import tensorflow as tf
tf.enable_eager_execution()
import sys, time

from tester import Tester
from option import args
from model import Model

model = Model()
tester = Tester(args, model)
tester.load_model()
tester.test()
