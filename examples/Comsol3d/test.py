from hpctool import ComsolWorker
from Comsol3d import *

model = MyModel()
myWorker = ComsolWorker(model, file_path,
                                           mph_options={'classkit': True},
                                           client_kwargs={'cores': 1,
                                                          'version': '5.5'})


myWorker.start()
myWorker.model.pre_build(x=[1, 0, 1, 0])
myWorker.model.build()
myWorker.model.save(dirname + '/1.mph')
