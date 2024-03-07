from string import printable
import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

def fetch_image(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.content
    else:
        print("Failed to fetch image. Status code:", response.status_code)


def crop_sides(image, left, top, right, bottom):
    width, height = image.size
    new_dimensions = (left, top, width - right, height - bottom)
    return image.crop(new_dimensions)


def save_resized_image(image_content, size=(256, 256), crop=0):
    if image_content is not None:
        # Open the image using BytesIO object as file-like object
        image = Image.open(BytesIO(image_content))
        
        # Resize the image
        cropped_image = crop_sides(image, crop, crop, crop, crop)
        resized_image = cropped_image.resize(size)
        
        return resized_image
    else:
        print("No image to resize and save.")


def batch_download(url, num_images, output_folder, offset_image_index=0, size=(128, 128), crop=32):
    for i in range(num_images):
        try:
            image_content = fetch_image(url)
            img = save_resized_image(image_content, size, crop)
            output_path = f"{output_folder}/image_{offset_image_index+i+1}.jpg"
            img.save(output_path)
            print(f"Image saved: {i+1}")
        
        except Exception as e:
            print(f"An error occurred: {e}")


if __name__ == "__main__":
    url = "https://thispersondoesnotexist.com"
    # output_filename = "person.jpg"
    # image_content = fetch_image(url)
    # img = save_resized_image(image_content, output_filename, (128, 128), 32)
    # img.save(output_filename)
    # plt.imshow(img)
    # plt.axis('off')
    # plt.show()
    batch_download(url, 1000, "images", 0)
