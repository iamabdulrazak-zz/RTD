import cv2, time
import numpy as np

# Loading the algorithm
nNetwork = cv2.dnn.readNet("model/yolov3-tiny.weights", "cfg/yolov3-tiny.cfg")
# loading the objects
objects = []
with open("dataset.names", "r") as f:
    objects = [line.strip() for line in f.readlines()]
print("List Of Objects:", objects)

# config the nNetwork object
layer_names = nNetwork.getLayerNames()
output_layers = [layer_names[i[0]-1] for i in nNetwork.getUnconnectedOutLayers()]

# defining color for each detected object
colors = np.random.uniform(0, 255, size=(len(objects), 3))

# loading, resizing the video and executing the frames
vid_in = cv2.VideoCapture('vid/cars.mp4') #to take from device camera (0, cv2.CAP_DSHOW)
# define the codec and create vid_out
vid_codec = cv2.VideoWriter_fourcc(*'mp4v')
vid_out = cv2.VideoWriter("vid/vid-detected/out.mp4", vid_codec, 20.0, (640, 480))
# detected frames in a second
st_time = time.time()  # counting time
frame_id = 0  # counting frames
# defining fonts
font = cv2.FONT_HERSHEY_PLAIN
while True:
	_,frame = vid_in.read()
	frame_id += 1
	height, width, channels = frame.shape
	vid_out.write(frame)

	# detecting and extracting features from the frame
	blob = cv2.dnn.blobFromImage(frame, 0.00392, (320, 320), (0, 0, 0), True, crop=False)

	# processing the frame by the algorithm
	nNetwork.setInput(blob)
	out = nNetwork.forward(output_layers)
	# showing information's
	class_ids = []
	confs = []
	boxs = []
	for outs in out:
		for detect in outs:
			score = detect[5:]
			class_id = np.argmax(score)
			conf = score[class_id]
			if conf > 0.5:
				# object detection and its coordinates
				center_x = int(detect[0] * width)
				center_y = int(detect[1] * height)
				w = int(detect[2] * width)
				h = int(detect[3] * height)

				# rec coordinates
				x = int(center_x - w / 2)
				y = int(center_y - h / 2)

				# all rec areas
				boxs.append([x, y, w, h])
				# how conf its of the detection
				confs.append(float(conf))
				# to know the name of the object detected
				class_ids.append(class_id)

	# looking through the objects
	objects_detected = len(boxs)
	print("Number of Objects detected:", objects_detected)
	# removing the double boxes/noise
	indexs = cv2.dnn.NMSBoxes(boxs, confs, 0.5, 0.4)
	print("Number of Boxs:", indexs)

	# calculating how much time past (FPS)
	pt_time = time.time() - st_time
	fps = frame_id / pt_time
	cv2.putText(frame, "FPS: " + str(round(fps, 2)), (10, 50), font, 2, (0, 0, 0), 2)

	for i in range(len(boxs)):
		if i in indexs:
			x, y, w, h = boxs[i]
			label = objects[class_ids[i]]
			conf = confs[i]
			color = colors[class_ids[i]]
			cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3)
			# put text to the frame
			cv2.putText(frame, label + " " + str(round(conf, 2)), (x+10, y+30), font, 2, (0, 0, 0), 2)
			print("FPS: "+ str(round(fps, 2)) + " | " + "Detected:", label + " | " + "Accuracy:", round(conf, 4))

	# displaying
	cv2.imshow("whoami 0.1", frame)
	# detecting the key to exit the video
	key = cv2.waitKey(1)
	if key == 27:
		break
# when everything done, release the video capture object
vid_in.release()
vid_out.release()
cv2.destroyAllWindows()
