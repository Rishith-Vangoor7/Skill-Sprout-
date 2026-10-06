import sys
from PIL import Image

def process_image(input_path, output_path):
    img = Image.open(input_path).convert("RGBA")
    width, height = img.size
    
    # Crop the bottom 25% to remove the text
    crop_height = int(height * 0.75)
    img = img.crop((0, 0, width, crop_height))
    
    # Get the background color from the top-left pixel
    bg_color = img.getpixel((0, 0))
    
    # Make background pixels transparent (with some tolerance)
    data = img.getdata()
    new_data = []
    
    tolerance = 15
    for item in data:
        # Check if color is close to background color
        if (abs(item[0] - bg_color[0]) <= tolerance and
            abs(item[1] - bg_color[1]) <= tolerance and
            abs(item[2] - bg_color[2]) <= tolerance):
            # Change to transparent
            new_data.append((255, 255, 255, 0))
        else:
            new_data.append(item)
            
    img.putdata(new_data)
    
    # Auto-crop the transparent borders
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)
        
    img.save(output_path, "PNG")

if __name__ == "__main__":
    process_image(sys.argv[1], sys.argv[2])
