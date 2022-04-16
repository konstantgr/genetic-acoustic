from hpctool import ComsolWorker
from Comsol3d import *

model = MyModel()
myWorker = ComsolWorker(model, file_path, mph_options={'classkit': True}, client_kwargs={'cores': 1, 'version': '5.5'})

myWorker.start()

cylinders_length, cylinders_radii, cylinders_separations = [100,200,300], [200,200,200], [0, 0, 0]
myWorker.model.pre_build(cylinders_length, cylinders_radii, cylinders_separations)
myWorker.model.build()

myWorker.model.export_image(model.geometry, dirname + '/u.png', props={'sourcetype': 'geometry'})
#  geometry drawing
myWorker.model.export_image("/GeomList/geom1/GeomFeatureList/wp1/sequence2D", dirname + '/u1.png', props={'sourcetype': 'other'})

myWorker.model.save(dirname + '/1.mph')
