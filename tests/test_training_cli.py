"""Actual argparse control contract, independent of Torch/GPU execution."""
import argparse
import ast
from contextlib import redirect_stderr
import io
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]


class TrainingCLI(unittest.TestCase):
    def test_explicit_cuda_false_and_default_true(self):
        source=ROOT/'train.py';tree=ast.parse(source.read_text())
        helper=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='parse_bool')
        main=next(node for node in tree.body if isinstance(node,ast.If))
        # Execute the actual -c argument declaration and its real type helper;
        # do not reconstruct its keyword wiring or import the model engine.
        argument=next(node for node in main.body if isinstance(node,ast.Expr)
                      and isinstance(node.value,ast.Call)
                      and isinstance(node.value.func,ast.Attribute)
                      and node.value.func.attr=='add_argument' and node.value.args
                      and isinstance(node.value.args[0],ast.Constant)
                      and node.value.args[0].value=='-c')
        namespace={'argparse':argparse,'parser':argparse.ArgumentParser(add_help=False)}
        exec(compile(ast.Module(body=[helper,argument],type_ignores=[]),str(source),'exec'),namespace)
        parser=namespace['parser']
        self.assertIs(parser.parse_args([]).cuda,True)
        for value in ('False','false','FALSE','0','no','off'):
            with self.subTest(value=value):self.assertIs(parser.parse_args(['-c',value]).cuda,False)
        for value in ('True','true','TRUE','1','yes','on'):
            with self.subTest(value=value):self.assertIs(parser.parse_args(['--cuda',value]).cuda,True)
        with redirect_stderr(io.StringIO()),self.assertRaises(SystemExit) as error:
            parser.parse_args(['-c','maybe'])
        self.assertEqual(error.exception.code,2)


if __name__=='__main__':unittest.main()
