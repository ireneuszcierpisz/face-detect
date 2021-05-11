"""
    Simple people detection application for testing packaging app openvino deployment tools.
    We can use the Deployment Manager present in OpenVINO to create a runtime package from our application. 
    To do this try the following:
        Start the Deployment Manager in interactive mode
        Select the hardware where you want to deploy your model
        Select the folder containing your application code, models, and data
    These packages can be easily sent to other hardware devices to be deployed.
    To deploy the Inference Engine components from the development machine to the target host, perform the following steps:
        Transfer the generated archive to the target host using your preferred method.
        Unpack the archive into the destination directory on the target host (replace the openvino_deployment_package with the name you use).
            For Linux:  tar xf openvino_deployment_package.tar.gz -C <destination_dir>
            For Windows, use an archiver your prefer.
"""

from openvino.inference_engine import IENetwork, IECore
import numpy as np
import cv2
import argparse
import logging

logging.getLogger().setLevel(logging.INFO)

def load_model(args):
    model = args.model
    device = args.device
    model_weights = model + '.bin'
    model_structure = model + '.xml'
    #net = IENetwork(model=model_structure, weights=model_weights)
    # face_detect.py:29: DeprecationWarning: 
    # Reading network using constructor is deprecated. 
    # Please, use IECore.read_network() method instead
    ie = IECore()
    logging.info('IE Plugin initialized.')
    net = ie.read_network(model=model_structure, weights=model_weights)
    print('NET.OUTPUTS:\n', net.outputs)
    #print(net.output_info)
    
    #print("Batch size:", net.batch_size)
    net.batch_size = 1
    print("Batch size:", net.batch_size)
    
    exec_net = ie.load_network(network=net, device_name=device, num_requests=1)
    logging.info('IENetwork loaded into the Plugin as an exec_net')
    
    #input_blob = next(iter(net.inputs))
    # face_detect.py:38: DeprecationWarning: 
    # 'inputs' property of IENetwork class is deprecated. 
    # To access DataPtrs user need to use 'input_data' property of InputInfoPtr objects 
    #which can be accessed by 'input_info' property.
    input_layer = next(iter(net.input_info))
    input_blob = input_layer
  
    output_blob = next(iter(net.outputs))
    return net, exec_net, input_blob, output_blob

def preprocess_frame(frame, net, input_blob):
    # DeprecationWarning: 'inputs' property of IENetwork class is deprecated. 
    # To access DataPtrs user need to use 'input_data' property of InputInfoPtr objects 
    # which can be accessed by 'input_info' property.
    #model_shape = net.inputs[input_blob].shape
    n, c, h, w = net.input_info[input_blob].input_data.shape
    #model_w = model_shape[3]
    #model_h = model_shape[2]
    frame4infer = np.copy(frame)
    #frame4infer = cv2.resize(frame4infer, (model_w, model_h))
    frame4infer = cv2.resize(frame4infer, (w, h))    
    frame4infer = frame4infer.transpose((2,0,1))
    #frame4infer = frame4infer.reshape(1, 3, model_h, model_w)
    frame4infer = frame4infer.reshape(n, c, h, w)
    
                #net.reshape({input_blob: (n, c, h, w)})
    
    return frame4infer

def detect_face(exec_net, frame4infer, input_blob, output_blob):
    exec_net.start_async(request_id=0, inputs={input_blob:frame4infer})
    if exec_net.requests[0].wait(-1) == 0:
    
    # DeprecationWarning: 'outputs' property of InferRequest is deprecated. 
    # Please instead use 'output_blobs' property.
        output = exec_net.requests[0].outputs[output_blob]
        #output = exec_net.requests[0].output_blobs['detection_out']
    return output

def find_bb_coord(output, height, width):
    bb_coordinates = ()
    conf = 0
    for box in output[0][0]: # Output shape is 1x1xNx7                
        confidence = box[2]
        if confidence >= 0.5:
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            
            # if multiple people in the same input frame
            # choose the one detected with bigest confidence
            if confidence > conf:
                conf = confidence
                bb_coordinates = (xmin,ymin,xmax,ymax)
            
    # if not face detected return coordinates to draw big bounding box 
    if conf == 0:
        bb_coordinates = (0, 0, width, height)
    return bb_coordinates

#def draw_bb(bb_coords, frame):
#    cv2.rectangle(frame, (bb_coords[0],bb_coords[1]), (bb_coords[2],bb_coords[3]), (255,0,0), 2)

def stream_cam(args, n):
    cam = cv2.VideoCapture(n)
    width = int(cam.get(3))
    height = int(cam.get(4))
    cam.open(n)
    net, exec_net, input_blob, output_blob = load_model(args)
    logging.info('Face detection DNN Model loaded! Starting inference...')
    try:
        if not cam.isOpened():
            print("Unable to open camera!")
        if cam.isOpened():
            print("Streaming from build-in camera!  Press 'q' or 'Ctrl+c' to leave.")
            while cam.isOpened():
                flag, frame = cam.read()
                if not flag:
                    break
                frame4infer = preprocess_frame(frame, net, input_blob)
                infer_output = detect_face(exec_net, frame4infer, input_blob, output_blob)
                bb_coords = find_bb_coord(infer_output, height, width)
                frame_copy = frame.copy()
                #cv2.rectangle(frame_copy, (bb_coords[0],bb_coords[1]), (bb_coords[2],bb_coords[3]), (255,0,0), 2)
                xc = (bb_coords[2] - bb_coords[0])//2 + bb_coords[0]
                yc = (bb_coords[3] - bb_coords[1])//2 + bb_coords[1]
                cv2.circle(frame_copy, (xc, yc), (yc - bb_coords[1]), (255,0,0), 1)
                
                cv2.ellipse(frame_copy, (xc, bb_coords[1]-(bb_coords[2] - bb_coords[0])//4), ((bb_coords[2] - bb_coords[0])//2, (bb_coords[2] - bb_coords[0])//5), 10, 0, 360, (0,255,255), 2)
                cv2.putText(frame_copy, "Copyrights: ICST Ltd.  ;)", (10,20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                if bb_coords[0] == 0 and bb_coords[2] == width:
                    cv2.putText(frame_copy, "No face detected!", (int(width/4), int(height/2)), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('The Face', frame_copy)
                k = cv2.waitKey(1) & 0xFF
                if k == ord('q'):
                    break
    except KeyboardInterrupt:
        pass

    cam.release()
    cv2.destroyAllWindows()


def main(args):
    stream_cam(args, args.cam)


if __name__=='__main__':
    print('Hello! Starting inference...')
    parser=argparse.ArgumentParser()
    logging.info('Get arguments.')
    parser.add_argument('--device', type=str, help='device', default='CPU')
    parser.add_argument('--model', type=str, help='The path to the model xml file', default='intel/face-detection-adas-0001/FP16/face-detection-adas-0001')
    parser.add_argument('--cam', type=int, help='The camera number: 0-10 where 0 is for laptop integrated cam, 1 is for first external cam.', default=0)
    args=parser.parse_args()

    main(args)
    