"""Optional real DRNet3D CPU train-loop check; no measured-data accuracy claim."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

HAS_TORCH = importlib.util.find_spec('torch') is not None


@unittest.skipUnless(HAS_TORCH, 'optional torch CPU integration dependency')
class TrainingLoop(unittest.TestCase):
    def test_real_network_training_and_heldout_batchnorm_state(self):
        import numpy as np
        import torch
        from train import train, loss_fn
        from networks.DepthRefinement.model import DRNet3D, DEPTH_VAL
        torch.set_num_threads(1)
        torch.manual_seed(42)
        model=DRNet3D(DEPTH_VAL,segment=True)
        optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        def batch(n):
            return {'radar':torch.randn(n,128,8,8),'lidar':torch.full((n,8,8),2.),
                    'lidar_mask':torch.stack([torch.ones(n,8,8),torch.zeros(n,8,8)],dim=1)}
        training=[batch(2)]
        heldout=[batch(1),batch(2)]
        before={name:value.clone() for name,value in model.named_parameters()}
        with tempfile.TemporaryDirectory() as output:
            train_loss,test_loss=train(model,optimizer,scheduler,training,heldout,
                                       torch.device('cpu'),2,loss_fn,Path(output),plot=False)
            self.assertTrue((Path(output)/'1.pth').is_file())
        self.assertTrue(np.isfinite(train_loss).all() and np.isfinite(test_loss).all())
        self.assertTrue(any(not torch.equal(before[name],value) for name,value in model.named_parameters()))
        # Exactly two training batches, no eval-time BN updates across four heldout batches.
        counters=[int(v) for name,v in model.named_buffers() if name.endswith('num_batches_tracked')]
        self.assertTrue(counters and all(value==2 for value in counters))
        self.assertFalse(model.training)
        self.assertEqual(scheduler.last_epoch,2)

    def test_dataset_rejects_equal_counts_with_different_stems(self):
        import numpy as np
        from dataloader.dataset import Dataset
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'radar').mkdir();(root/'lidar').mkdir()
            np.savez(root/'radar/000.npz',intensity=np.zeros((128,16,128)),velocity=np.zeros((128,16,128)))
            np.save(root/'lidar/001.npy',np.zeros((16,128)))
            with self.assertRaisesRegex(ValueError,'matching nonempty frame stems'):Dataset(root)


if __name__=='__main__':unittest.main()
