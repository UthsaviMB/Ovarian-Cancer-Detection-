from flask import Flask, render_template, url_for, request
import sqlite3
import os
import numpy as np
from tensorflow.keras.models import load_model
import pickle
import shutil
import cv2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import  img_to_array , array_to_img
import numpy as np  # dealing with arrays

from random import shuffle  # mixing up or currently ordered data that might lead our network astray in training.

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array, array_to_img
from keras.preprocessing import image
from skimage.filters import laplace, sobel
from skimage import exposure, img_as_float






# Load the trained model
model = load_model('ResNet50_model.h5')

# Load class names from the pickle file
with open('class_names.pkl', 'rb') as f:
    class_names = pickle.load(f)

def predict_image(image):
    img =load_img(image, target_size=(150, 150))
    img_array =img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction)
    print(predicted_class_index)
    predicted_class = class_names[predicted_class_index]
    print("predicted_class:",predicted_class)
    prediction1 = prediction.tolist()
    print(prediction1[0][predicted_class_index]*100)
    return predicted_class, prediction1[0][predicted_class_index]*100

connection = sqlite3.connect('user_data.db')
cursor = connection.cursor()

command = """CREATE TABLE IF NOT EXISTS user(name TEXT, password TEXT, mobile TEXT, email TEXT)"""
cursor.execute(command)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/userlog', methods=['GET', 'POST'])
def userlog():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']

        query = "SELECT name, password FROM user WHERE name = '"+name+"' AND password= '"+password+"'"
        cursor.execute(query)

        result = cursor.fetchall()

        if len(result)==0:
            return render_template('index.html',msg='Sorry, Incorrect Credentials Provided,  Try Again')
        else:
            return render_template('userlog.html')

    return render_template('home.html')


@app.route('/userreg', methods=['GET', 'POST'])
def userreg():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']
        mobile = request.form['phone']
        email = request.form['email']
        
        print(name, mobile, email, password)

        command = """CREATE TABLE IF NOT EXISTS user(name TEXT, password TEXT, mobile TEXT, email TEXT)"""
        cursor.execute(command)

        cursor.execute("INSERT INTO user VALUES ('"+name+"', '"+password+"', '"+mobile+"', '"+email+"')")
        connection.commit()

        return render_template('index.html', msg='Successfully Registered')
    
    return render_template('index.html')



@app.route('/userlog.html', methods=['GET'])
def indexBt():
      return render_template('userlog.html')




@app.route('/graph.html', methods=['GET', 'POST'])
def graph():
    
    images = ['http://127.0.0.1:5000/static/Accu_plt.png',
              
              'http://127.0.0.1:5000/static/loss_plt.png']
    content=['Accuracy Graph',
            ' Loss Graph']

            
    
        
    return render_template('graph.html',images=images,content=content)
    



@app.route('/image', methods=['GET', 'POST'])
def image():
    if request.method == 'POST':
        
        dirPath = "static/images"
        fileList = os.listdir(dirPath)
        for fileName in fileList:
            os.remove(dirPath + "/" + fileName)
        fileName=request.form['filename']
        dst = "static/images"
        
        shutil.copy("test/"+fileName, dst)
        image = cv2.imread("test/"+fileName)
        #color conversion
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
       # 1. RGB to Grayscale conversion (Luminosity method)
        luminosity_gray = 0.21 * image[:, :, 2] + 0.72 * image[:, :, 1] + 0.07 * image[:, :, 0]
        cv2.imwrite('static/luminosity_gray.jpg', luminosity_gray)
        # 2. Noise removal methods

        # Median filter
        median_filtered = cv2.medianBlur(gray_image, 5)
        cv2.imwrite('static/median_filtered.jpg', median_filtered)

        # Gaussian filter
        gaussian_filtered = cv2.GaussianBlur(gray_image, (5, 5), 1)
        cv2.imwrite('static/gaussian_filtered.jpg', gaussian_filtered)

        # High pass filter (Laplacian filter to enhance edges)
        high_pass = cv2.Laplacian(gray_image, cv2.CV_64F)
        cv2.imwrite('static/high_pass_filtered.jpg', high_pass)
        # 3. Image sharpening

        # Unsharp masking
        gaussian = cv2.GaussianBlur(gray_image, (9, 9), 10.0)
        unsharp_image = cv2.addWeighted(gray_image, 1.5, gaussian, -0.5, 0)
        cv2.imwrite('static/unsharp_masked.jpg', unsharp_image)

        # Laplacian filter for sharpening
        laplacian = laplace(gray_image)
        laplacian_image = gray_image - laplacian * 255
        cv2.imwrite('static/laplacian_sharpened.jpg', laplacian_image)

        # Sobel filter for edge detection and sharpening
        sobel_x = sobel(gray_image)
        cv2.imwrite('static/sobel_edge.jpg', sobel_x * 255)

        # 4. Thresholding

        # Gaussian thresholding
        _, gaussian_threshold = cv2.threshold(gaussian_filtered, 127, 255, cv2.THRESH_BINARY)
        cv2.imwrite('static/gaussian_thresholded.jpg', gaussian_threshold)

        # Adaptive thresholding
        adaptive_threshold = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                cv2.THRESH_BINARY, 11, 2)
        cv2.imwrite('static/adaptive_thresholded.jpg', adaptive_threshold)

        # 5. Segmentation

        # Edge-based segmentation (Canny edge detection)
        edges = cv2.Canny(gray_image, 100, 200)
        cv2.imwrite('static/edge_based_segmentation.jpg', edges)

        # Region-based segmentation (Watershed algorithm)
        # First, convert to binary image and then apply watershed
        ret, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # Noise removal
        kernel = np.ones((3,3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

        # Sure background area
        sure_bg = cv2.dilate(opening, kernel, iterations=3)

        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        ret, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)

        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)

        # Marker labeling
        ret, markers = cv2.connectedComponents(sure_fg)

        # Add one to all labels so that sure background is not 0, but 1
        markers = markers + 1

        # Now, mark the region of unknown with zero
        markers[unknown == 255] = 0

        # Apply watershed
        markers = cv2.watershed(image, markers)
        image[markers == -1] = [255, 0, 0]  # Mark boundaries with red color

        cv2.imwrite('static/region_based_segmentation.jpg', image)

        # White areas represent tumor/lymph nodes/metastasis regions
        image = cv2.imread("test/"+fileName, cv2.IMREAD_GRAYSCALE)



       # Apply Gaussian Blur and thresholding
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        _, binary_mask = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        # Find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Collect contour areas for dynamic thresholding
        areas = [cv2.contourArea(contour) for contour in contours if cv2.contourArea(contour) > 0]

        # Calculate dynamic thresholds based on percentiles
        tumor_threshold = np.percentile(areas, 75) if areas else 0
        lymph_node_threshold = np.percentile(areas, 50) if areas else 0
        metastasis_threshold = np.percentile(areas, 25) if areas else 0

        # Initialize counters and grading scores
        tumor_count = 0
        lymph_node_count = 0
        metastasis_count = 0
        grade_score = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            
            circularity = 4 * np.pi * (area / (perimeter * perimeter)) if perimeter > 0 else 0
            solidity = area / hull_area if hull_area > 0 else 0

            # Determine TNM category based on dynamic thresholds
            if area > tumor_threshold and circularity > 0.7:
                tumor_count += 1
                label = "Tumor"
            elif lymph_node_threshold < area <= tumor_threshold and circularity > 0.5:
                lymph_node_count += 1
                label = "Lymph Node"
            elif area <= metastasis_threshold:
                metastasis_count += 1
                label = "Metastasis"
            else:
                continue

            # Grade scoring based on solidity and circularity
            if solidity > 0.8 and circularity > 0.7:
                grade = "Low Grade (Well-Organized)"
            elif 0.5 < solidity <= 0.8 and 0.5 < circularity <= 0.7:
                grade = "Intermediate Grade (Moderately Organized)"
            else:
                grade = "High Grade (Disorganized/Solid Growth)"
            grade_score += (1 if "Low Grade" in grade else 2 if "Intermediate" in grade else 3)

            # Draw and label contours
            x, y, w, h = cv2.boundingRect(contour)
            #cv2.putText(image, f"{label}: {grade}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)

        # Calculate the overall grade score average
        total_cells = tumor_count + lymph_node_count + metastasis_count
        average_grade_score = grade_score / total_cells if total_cells > 0 else 0

        # Print the counts and grading score
        print("Tumor Count:", tumor_count)
        print("Lymph Node Count:", lymph_node_count)
        print("Metastasis Count:", metastasis_count)
        print("Average Grade Score:", grade)



        predicted_class, accuracy = predict_image("test/"+fileName)
        print("Predicted class:", predicted_class)
        print("Accuracy is:", accuracy)
       
        f = open('acc.txt', 'w')
        f.write(str(accuracy))
        f.close()

        
       
        str_label=""
        accuracy=""
        if predicted_class =="CC":
            str_label="CC(Clear-Cell Ovarian Carcinoma)"
            
            

        elif predicted_class =="EC":
            str_label="EC(Endometrioid)"


        elif predicted_class =="HGSC":
            str_label="HGSC(High-Grade Serous Carcinoma)"

        elif predicted_class =="LGSC":
            str_label="LGSC(Low-Grade Serous)"

        elif predicted_class =="MC":
            str_label="MC(Mucinous Carcinoma)"

        A=(predicted_class=='CC')
        B=(predicted_class=='EC')
        C=(predicted_class=='HGSC')
        D=(predicted_class=='LGSC')
        E=(predicted_class=='MC')
    
        dic={'CC':A,'EC':B,'HGSC':C,'LGSC':D,'MC':E}
        algm = list(dic.keys()) 
        accu = list(dic.values()) 
        fig = plt.figure(figsize = (5, 5))  
        plt.bar(algm, accu, color ='maroon', width = 0.3)  
        plt.xlabel("Comparision") 
        plt.ylabel("Accuracy Level") 
        plt.title("Accuracy Comparision between \n Ovarian Cancer Detection")
        plt.savefig('static/matrix.png')


        
            

       

       
        f = open('acc.txt', 'r')
        accuracy = f.read()
        f.close()
        print(accuracy)
        

        # Print results
        

        print("Tumor Count:", tumor_count)
        print("Lymph Node Count:", lymph_node_count)
        print("Metastasis Count:", metastasis_count)
        print("Cancer Grade:", grade)

        cell_count = [tumor_count, lymph_node_count, metastasis_count,grade]

        ImageDisplay=["http://127.0.0.1:5000/static/images/"+fileName,
        "http://127.0.0.1:5000/static/luminosity_gray.jpg",
        "http://127.0.0.1:5000/static/median_filtered.jpg",
        "http://127.0.0.1:5000/static/gaussian_filtered.jpg",
        "http://127.0.0.1:5000/static/high_pass_filtered.jpg",
        "http://127.0.0.1:5000/static/unsharp_masked.jpg",
        "http://127.0.0.1:5000/static/laplacian_sharpened.jpg",
        "http://127.0.0.1:5000/static/sobel_edge.jpg",
        "http://127.0.0.1:5000/static/gaussian_thresholded.jpg",
        "http://127.0.0.1:5000/static/adaptive_thresholded.jpg",
        "http://127.0.0.1:5000/static/edge_based_segmentation.jpg",
        "http://127.0.0.1:5000/static/region_based_segmentation.jpg",
        "http://127.0.0.1:5000/static/matrix.png"]

        labels = ['Original', 'Gray Image',
       'Median Filter', 'Gaussian Filter', 'High Pass Filter',
        'Unsharp Masking', 'Laplacian', 'Sobel',
        'Gaussian Threshold', 'Adaptive Threshold',
        'Edge Based (Canny)', 'Region Based (Watershed)' , 'graph']

        return render_template('results.html', labels=labels, status=str_label,accuracy=accuracy,cell_count=cell_count,grade=grade,ImageDisplay=ImageDisplay,n=len(ImageDisplay))
        
    return render_template('index.html')



        
        
        
       




@app.route('/logout')
def logout():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
