import cv2
import numpy as np
import tensorflow as tf
from plate_finder import PlateFinder

class OCR:
    def __init__(self, modelFile, labelFile):
        self.model_file = modelFile
        self.label_file = labelFile
        self.label = self.load_label(self.label_file)
        self.graph = self.load_graph(self.model_file)
        self.sess = tf.compat.v1.Session(graph=self.graph, config=tf.compat.v1.ConfigProto())
        
    def load_graph(self, modelFile):
        graph = tf.Graph()
        graph_def = tf.compat.v1.GraphDef()
        with open(modelFile, "rb") as f:
            graph_def.ParseFromString(f.read())
        with graph.as_default():
            tf.import_graph_def(graph_def)
        return graph
        
    def load_label(self, labelFile):
        label = []
        with tf.io.gfile.GFile(labelFile) as f:
            for l in f.readlines():
                label.append(l.rstrip())
        return label
        
    def convert_tensor(self, image, imageSizeOutput):
        image = cv2.resize(image, dsize=(imageSizeOutput, imageSizeOutput), interpolation=cv2.INTER_CUBIC)
        np_image_data = np.asarray(image)
        np_image_data = cv2.normalize(np_image_data.astype('float'), None, -0.5, .5, cv2.NORM_MINMAX)
        return np.expand_dims(np_image_data, axis=0)
        
    def label_image(self, tensor):
        input_operation = self.graph.get_operation_by_name("import/input")
        output_operation = self.graph.get_operation_by_name("import/final_result")
        results = self.sess.run(output_operation.outputs[0], {input_operation.outputs[0]: tensor})
        results = np.squeeze(results)
        top = results.argsort()[-1:][::-1]
        return self.label[top[0]]
        
    def label_image_list(self, listImages, imageSizeOutput):
        plate = ""
        for img in listImages:
            if cv2.waitKey(25) & 0xFF == ord('q'): break
            plate = plate + self.label_image(self.convert_tensor(img, imageSizeOutput))
        return plate, len(plate)

if __name__ == "__main__":
    findPlate = PlateFinder(minPlateArea=4100, maxPlateArea=15000)
    # Ensure your .pb and .txt model files are placed inside the /model folder!
    model = OCR(modelFile="model/binary_128_0.50_ver3.pb", labelFile="model/binary_128_0.50_labels_ver2.txt")
    
    cap = cv2.VideoCapture('test.MOV') # Replace with 0 for webcam live view
    while (cap.isOpened()):
        ret, img = cap.read()
        if not ret: break
        
        cv2.imshow('Original Video', img)
        if cv2.waitKey(25) & 0xFF == ord('q'): break
        
        possible_plates = findPlate.find_possible_plates(img)
        if possible_plates is not None:
            for i, p in enumerate(possible_plates):
                chars_on_plate = findPlate.char_on_plate[i]
                recognized_plate, _ = model.label_image_list(chars_on_plate, imageSizeOutput=128)
                print(f"Detected Plate Number: {recognized_plate}")
                cv2.imshow('Extracted Plate Area', p)
                if cv2.waitKey(25) & 0xFF == ord('q'): break
                
    cap.release()
    cv2.destroyAllWindows()