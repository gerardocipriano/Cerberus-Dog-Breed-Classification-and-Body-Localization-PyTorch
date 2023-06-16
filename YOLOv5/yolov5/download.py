

from roboflow import Roboflow
rf = Roboflow(api_key="ewoNsPmelTaDzAgol6iU")
project = rf.workspace("unibo-hheit").project("cerberus")
dataset = project.version(9).download("yolov5")
