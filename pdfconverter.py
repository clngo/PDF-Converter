import cv2
import pytesseract
import glob


# Returns a list of files in the folder. 
def get_files():
    files = glob.glob("*.png")
    return files


# Image to string (text)
def ocr_core(img):
    text = pytesseract.image_to_string(img)
    return text


# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# noise removal
def remove_noise(image):
    return cv2.medianBlur(image, 5)


# thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def scanning_representation(image, config):
    # https://youtu.be/PY_N1XdFp4w
    
    height, width, _ = image.shape

    boxes = pytesseract.image_to_boxes(image, config=config)
    for box in boxes.splitlines():
        box = box.split(" ")
        green = (0,255,0)
        image = cv2.rectangle(image, (int(box[1]), height - int(box[2])), (int(box[3]), height - int(box[4])), green, 2)

    if height > 980 or width > 1820:
        hratio = 980 / height
        new_height = round(height * hratio)
        new_width = round(width * hratio)
        
        if new_width > 1820: 
            wratio = 1820 / width
            new_height = round(height * wratio)
            new_width = round(width * wratio)

        image = cv2.resize(image, (new_width, new_height))

    cv2.imshow("resultimg", image)
    cv2.waitKey(0)
    

def data(image, config):
    data = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DICT)
    num_boxes = len(data["text"])
    green = (0,255,0)

    for i in range(num_boxes):
        if float(data["conf"][i]) > 80:
            (x, y, width, height) = (data["left"][i], data["top"][i], data["width"][i], data["height"][i])
            image = cv2.rectangle(image, (x, y), (x+width, y+height), green, 2)
            image = cv2.putText(image, data['text'][i], (x, y+height+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, green, 2, cv2.LINE_AA)
    
    height, width, _ = image.shape

    if height > 980 or width > 1820:
        hratio = 980 / height
        new_height = round(height * hratio)
        new_width = round(width * hratio)
        print("h: {height} w: {width}")
        if new_width > 1820: 
            wratio = 1820 / width
            new_height = round(height * wratio)
            new_width = round(width * wratio)

        image = cv2.resize(image, (new_width, new_height))

    cv2.imshow("resultimg", image)
    cv2.waitKey(0)
    


def presenting_text(image, mode=None):
    text_print = image

    if mode == "y":
        image = get_grayscale(image)
        image = thresholding(image)
        image = remove_noise(image)
        text_print = image

    return ocr_core(text_print)


def main():

    list_of_png = get_files()

    output_file = open("output.txt", "a")
    myconfig = r"--psm 3 --oem 3"

    for png in list_of_png:
        img = cv2.imread(png)
    
        # Change mode to "y" to remove image noise
        text_print = presenting_text(img)
        print(png)
        print(text_print)

        output_file.write(f"\n{png}\n")
        output_file.write(text_print)
        # # Creating PDF
        # pdf = FPDF('P', 'mm', "Letter")

        # # Add a page
        # pdf.add_page()

        # # font
        # pdf.set_font("Courier", "", 16)

        # # Add text
        # pdf.cell(40, 10, f"{text_print}")

        # # Resulting pdf
        # pdf.output("pdfoutput.pdf")


    
    # Do NOT uncomment both

    # Merely shows what is being scanned
    scanning_representation(img, myconfig)

    # Includes text next to each box. Not as accurate as scanning_representation, but better for just visual image.
    # data(img, myconfig)


if __name__ == "__main__":
    main()

