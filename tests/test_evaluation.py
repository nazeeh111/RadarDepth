"""Actual CPU checkpoint→depth→geometry checks; synthetic weights/data, no accuracy claim."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HAS_TORCH = importlib.util.find_spec('torch') is not None
ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(HAS_TORCH, 'optional Torch CPU evaluation dependency')
class EvaluationTests(unittest.TestCase):
    def setUp(self):
        import torch
        torch.set_num_threads(1)
        torch.manual_seed(16)

    def test_actual_network_softmax_dtype_and_inference_state(self):
        import numpy as np
        import torch
        from eval import infer_frame
        from networks.DepthRefinement.model import DRNet3D, DEPTH_VAL
        model = DRNet3D(DEPTH_VAL).double().train()
        model.cost_regularization.conv0.bn.eval()
        original_modes = {name:module.training for name,module in model.named_modules()}
        radar = np.random.default_rng(16).normal(size=(128,8,8))
        before = {k:v.clone() for k,v in model.named_buffers()}
        observed=[]
        handle=model.register_forward_pre_hook(lambda module,args: observed.append((module.training,torch.is_grad_enabled(),torch.is_inference_mode_enabled())))
        depth, probability, selected, elapsed = infer_frame(model, radar)
        handle.remove()
        self.assertEqual(observed,[(False,False,True)])
        self.assertTrue(model.training)
        self.assertEqual({name:module.training for name,module in model.named_modules()},original_modes)
        for name,value in model.named_buffers(): torch.testing.assert_close(value,before[name],rtol=0,atol=0)
        model.eval()
        with torch.inference_mode(): result=model(torch.as_tensor(radar).unsqueeze(0).unsqueeze(0))
        logits=result['segment_mask'][0].numpy()
        self.assertEqual(logits.shape,(2,8,8))
        shifted=logits-logits.max(axis=0)
        expected=np.exp(shifted[0])/np.exp(shifted).sum(axis=0)
        np.testing.assert_allclose(probability,expected,rtol=1e-14,atol=1e-14)
        np.testing.assert_array_equal(depth,result['depth'][0].numpy())
        np.testing.assert_array_equal(selected,expected>=.4)
        self.assertGreater(elapsed,0)
        self.assertFalse(model(torch.as_tensor(radar).unsqueeze(0).unsqueeze(0))['depth'].is_leaf)

    def test_nonsegmented_model_and_invalid_inputs(self):
        import numpy as np
        from eval import infer_frame
        from networks.DepthRefinement.model import DRNet3D, DEPTH_VAL
        model=DRNet3D(DEPTH_VAL,segment=False)
        depth,probability,selected,_=infer_frame(model,np.ones((128,8,8)))
        np.testing.assert_array_equal(probability,np.ones_like(depth))
        self.assertTrue(selected.all())
        for invalid in [np.ones((127,8,8)),np.ones((128,7,8)),np.full((128,8,8),np.nan)]:
            with self.assertRaises(ValueError):infer_frame(model,invalid)
        with self.assertRaises(ValueError):infer_frame(model,np.ones((128,8,8)),threshold=1.1)

    def test_depth_loss_empty_returns_differentiable_zero(self):
        import torch
        from networks.DepthRefinement.network_parts import huber_loss
        prediction=torch.ones((1,4,4),requires_grad=True)
        labels=torch.full_like(prediction,10.)
        value=huber_loss(prediction,labels)
        self.assertEqual(float(value.detach()),0.)
        value.backward()
        torch.testing.assert_close(prediction.grad,torch.zeros_like(prediction))
        labels[0,0,:4]=torch.tensor([2.,0.,-1.,float('nan')])
        self.assertAlmostEqual(float(huber_loss(prediction,labels).detach()),.5)

    def test_raw_frame_order_and_metrics_have_independent_oracles(self):
        import numpy as np
        from eval import load_raw_lidar, depth_metrics
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            np.save(root/'000.npy',[[1.,2.,3.,8.]])
            np.save(root/'001.npy',[[4.,5.,6.,9.]])
            T=np.eye(4);T[:3,3]=[.2,-.3,.4]
            points=load_raw_lidar(root,T,['00001','00000'])
            np.testing.assert_allclose(points[0],[[4.2,4.7,6.4]])
            np.testing.assert_allclose(points[1],[[1.2,1.7,3.4]])
            with self.assertRaisesRegex(ValueError,'Missing raw LiDAR'):load_raw_lidar(root,T,['5'])
            np.save(root/'0000.npy',[[0.,0.,0.]])
            with self.assertRaisesRegex(ValueError,'duplicate numeric'):load_raw_lidar(root,T,['0'])
        depth=np.array([[3.,4.],[1.,5.]])
        labels=np.array([[2.,2.],[10.,0.]])
        metrics=depth_metrics(depth,labels,np.array([[True,False],[True,False]]))
        self.assertEqual(metrics['valid_depth_pixels'],2)
        self.assertAlmostEqual(metrics['depth_mae_m'],1.5)
        self.assertAlmostEqual(metrics['depth_rmse_m'],(2.5)**.5)
        self.assertEqual([metrics[k] for k in ['true_positive','false_positive','false_negative','true_negative']],[1,1,1,1])
        self.assertIsNone(depth_metrics(depth,np.full((2,2),10.),np.ones((2,2),bool))['depth_mae_m'])

    def test_bad_calibration_rejected_before_output_or_inference(self):
        import numpy as np
        from dataloader.dataset import Dataset
        from eval import process_dataset
        from networks.DepthRefinement.model import DRNet3D, DEPTH_VAL
        model=DRNet3D(DEPTH_VAL)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'radar').mkdir();(root/'lidar').mkdir()
            np.savez(root/'radar/000.npz',intensity=np.zeros((128,16,128)),velocity=np.zeros((128,16,128)))
            np.save(root/'lidar/000.npy',np.full((16,128),2.))
            dataset=Dataset(root)
            with np.load(ROOT/'calib/radarMapping.npz') as source:calib={k:source[k] for k in source.files}
            for index,(field,value) in enumerate([('azimuth_bins',np.full(128,np.nan)),('elevation_bins',np.full(32,np.inf)),('range_bin_width',.1),('azimuth_bins',np.linspace(4,5,128))]):
                with self.subTest(field=field,index=index):
                    path=root/f'calib{index}.npz';np.savez(path,**dict(calib,**{field:value}))
                    output=root/f'out{index}'
                    with self.assertRaises(ValueError):process_dataset(model,dataset,path,output_dir=output)
                    self.assertFalse(output.exists())
            np.savez(root/'radar/000.npz',intensity=np.zeros((128,8,8)),velocity=np.zeros((128,8,8)))
            calls=[]
            handle=model.register_forward_pre_hook(lambda module,args:calls.append(True))
            try:
                with self.assertRaisesRegex(ValueError,'calibrated angular grid'):
                    process_dataset(model,dataset,ROOT/'calib/radarMapping.npz',output_dir=root/'wrong-frame')
            finally:handle.remove()
            self.assertEqual(calls,[])
            self.assertFalse((root/'wrong-frame').exists())
            np.save(root/'lidar/000.npy',np.array([[2.,0.,-1.],[np.nan,10.,8.]]))
            masks=dataset[0]['lidar_mask']
            np.testing.assert_array_equal(masks[0],[[True,False,False],[False,False,False]])
            np.testing.assert_array_equal(masks.sum(axis=0),1)

    def test_train_checkpoint_public_cli_and_changed_input_geometry(self):
        import numpy as np
        import torch
        from train import train, loss_fn
        from networks.DepthRefinement.model import DRNet3D, DEPTH_VAL
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); dataset=root/'data';(dataset/'radar').mkdir(parents=True);(dataset/'lidar').mkdir()
            model=DRNet3D(DEPTH_VAL)
            optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
            scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
            radar=torch.randn(2,128,8,8)
            label=torch.full((2,8,8),2.)
            batch={'radar':radar,'lidar':label,'lidar_mask':torch.stack([torch.ones_like(label),torch.zeros_like(label)],dim=1)}
            # Real gradient update and state_dict save; this is a software fixture,
            # not a supplied trained research artifact or learned task evaluation.
            train(model,optimizer,scheduler,[batch],[batch],torch.device('cpu'),1,loss_fn,root,plot=False)
            volume=np.random.default_rng(92).normal(size=(128,16,128)).astype(np.float32)
            np.savez(dataset/'radar/00000.npz',intensity=volume,velocity=np.zeros_like(volume))
            truth=np.full((16,128),2.);truth[:2]=10
            np.save(dataset/'lidar/00000.npy',truth)
            np.savez(dataset/'radar/00001.npz',intensity=volume,velocity=np.zeros_like(volume))
            np.save(dataset/'lidar/00001.npy',truth)
            command=[sys.executable,str(ROOT/'radardepth.py'),'evaluate','-m',str(root/'0.pth'),'-c',str(ROOT/'calib/radarMapping.npz'),'-d',str(dataset),'--max-frames','1']
            first=root/'first';changed=root/'changed'
            record=subprocess.run(command+['-o',str(first)],cwd=root,capture_output=True,text=True,check=True)
            report=json.loads(record.stdout);self.assertEqual(len(report['frames']),1)
            self.assertEqual(set(report['frames'][0]['inputs_sha256']),{'00000.npz','00000.npy'})
            with np.load(first/'00000.npz') as frame:
                depth=frame['depth_m'];chosen=frame['selected_mask'];masked=frame['masked_depth_m']
                independent=float(np.abs(depth[truth<7.5896832]-truth[truth<7.5896832]).mean())
                self.assertAlmostEqual(report['frames'][0]['metrics']['depth_mae_m'],independent,places=12)
                with np.load(ROOT/'calib/radarMapping.npz') as calib:
                    e=calib['elevation_bins'][8:24,None];a=calib['azimuth_bins'][None,:]
                expected=np.stack([masked*np.cos(e)*np.cos(a),masked*np.cos(e)*np.sin(a),masked*np.sin(e)*np.ones_like(a)],axis=-1)[chosen]
                text=(first/'00000.ply').read_text().split('end_header\n')[1]
                actual=np.loadtxt(text.splitlines()).reshape(-1,3)
                np.testing.assert_allclose(actual,expected,rtol=1e-13,atol=1e-13)
                self.assertEqual(len(actual),report['frames'][0]['points'])
            np.savez(dataset/'radar/00000.npz',intensity=volume*2+1,velocity=np.zeros_like(volume))
            subprocess.run(command+['-o',str(changed)],cwd=root,capture_output=True,text=True,check=True)
            self.assertNotEqual((first/'00000.npz').read_bytes(),(changed/'00000.npz').read_bytes())
            before=(first/'evaluation.json').read_bytes()
            failed=subprocess.run(command+['-o',str(first)],cwd=root,capture_output=True,text=True)
            self.assertEqual(failed.returncode,2)
            self.assertEqual((first/'evaluation.json').read_bytes(),before)
            self.assertIn('File exists',failed.stderr)


if __name__=='__main__':unittest.main()
