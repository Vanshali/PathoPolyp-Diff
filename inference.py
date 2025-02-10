from diffusers import StableDiffusionPipeline
import torch

from diffusers import UNet2DConditionModel

# Download from https://huggingface.co/CompVis/stable-diffusion-v1-4/tree/main into "compv" folder.  
model_path = "./compv/"


unet = UNet2DConditionModel.from_pretrained("./result_dir/checkpoint/", subfolder="unet")  # Replace "./output_dir/checkpoint/" with your saved checkpoint path 

pipe = StableDiffusionPipeline.from_pretrained(model_path, unet=unet, torch_dtype=torch.float32, use_safetensors=True)
    

pipe.safety_checker = None
pipe.requires_safety_checker = False

pipe.to("cuda")


### Uncomment the sections below as per the requirement. 
#Section 1: colonoscopy image with polyp, good-quality, clear
#Section 2: colonoscopy image without polyp, good-quality, clear
#Section 3: colonoscopy image with polyp, good-quality, clear (negative prompt = "blur")
#Section 4: colonoscopy image without polyp, good-quality, clear (negative prompt = "blur")
#Section 5: colonoscopy image with polyp, good-quality, clear (negative prompt = "low-quality")
#Section 6: colonoscopy image without polyp, good-quality, clear (negative prompt = "low-quality")
#Section 7: colonoscopy image with polyp, good-quality, clear (negative prompt = "blur, low-quality")
#Section 8: colonoscopy image without polyp, good-quality, clear (negative prompt = "blur, low-quality")
#Section 9: colonoscopy image with adenomatous polyp, white light imaging, good-quality, clear
#Section 10: colonoscopy image with hyperplastic polyp, white light imaging, good-quality, clear
#Section 11: colonoscopy image with adenomatous polyp, narrow band imaging, good-quality, clear
#Section 12: colonoscopy image with hyperplastic polyp, narrow band imaging, good-quality, clear
#Section 13: With Weighted Control Mechanism (colonoscopy image with hyperplastic polyp, narrow band imaging, good-quality, clear)
#Section 14: With Weighted Control Mechanism (colonoscopy image with adenomatous polyp, narrow band imaging, good-quality, clear)

           
#Section 1:
#******************************* colonoscopy image with polyp, good-quality, clear***************************************#                
n_prompt = "blur, low-quality"
images = pipe(prompt="colonoscopy image with polyp, good-quality, clear", num_images_per_prompt=30).images  
for i, image in enumerate(images):
    image.save("./output_image_dir/polyp/"+str(i)+".png")
   

'''
#Section 2:
#*********************************** colonoscopy image without polyp, good-quality, clear ********************************************#
images = pipe(prompt="colonoscopy image without polyp, good-quality, clear", num_images_per_prompt=30).images  
for i, image in enumerate(images):
    image.save("./output_image_dir/non_polyp/"+str(i)+".png")

#Section 3:    
#***************************** colonoscopy image with polyp, good-quality, clear (negative prompt = "blur") ************************************************#
n_prompt = "blur"
images = pipe(prompt="colonoscopy image with polyp, good-quality, clear", negative_prompt= n_prompt, num_images_per_prompt=30).images  
for i, image in enumerate(images):
    image.save("./output_image_dir/polyp/neg_blur/"+str(i)+".png")

#Section 4:
#***************************** colonoscopy image without polyp, good-quality, clear (negative prompt = "blur") ************************************************#
n_prompt = "blur"
images = pipe(prompt="colonoscopy image without polyp, good-quality, clear", negative_prompt= n_prompt, num_images_per_prompt=30).images  
for i, image in enumerate(images):
    image.save("./output_image_dir/non_polyp/neg_blur/"+str(i)+".png")

#Section 5:    
#******************************* colonoscopy image with polyp, good-quality, clear (negative prompt = "low-quality")***************************************#        
n_prompt = "low-quality"
images = pipe(prompt="colonoscopy image with polyp, good-quality, clear", negative_prompt= n_prompt, num_images_per_prompt=30).images  
for i, image in enumerate(images):
    image.save("./output_image_dir/polyp/neg_low-quality/"+str(i)+".png")

#Section 6:
#******************************* colonoscopy image without polyp, good-quality, clear (negative prompt = "low-quality")***************************************#        
n_prompt = "low-quality"
images = pipe(prompt="colonoscopy image without polyp, good-quality, clear", negative_prompt= n_prompt, num_images_per_prompt=30).images  
for i, image in enumerate(images):
    image.save("./output_image_dir/non_polyp/neg_low-quality/"+str(i)+".png")
    
#Section 7:    
#******************************* colonoscopy image with polyp, good-quality, clear (negative prompt = "blur, low-quality")***************************************#                
n_prompt = "blur, low-quality"
images = pipe(prompt="colonoscopy image with polyp, good-quality, clear", negative_prompt= n_prompt, num_images_per_prompt=30).images  
for i, image in enumerate(images):
    image.save("./output_image_dir/polyp/neg_blur_low-quality/"+str(i)+".png")
        
#Section 8:        
#******************************* colonoscopy image without polyp, good-quality, clear (negative prompt = "blur, low-quality")***************************************#                
n_prompt = "blur, low-quality"
images = pipe(prompt="colonoscopy image without polyp, good-quality, clear", negative_prompt= n_prompt, num_images_per_prompt=30).images  
for i, image in enumerate(images):
    image.save("./output_image_dir/non_polyp/neg_blur_low-quality/"+str(i)+".png")
    
###############################################################################################################################################
##################################  Pathology based generation with different imaging modalities ##############################################       
###############################################################################################################################################   

n_prompt = "blur, low-quality"

#Section 9:
#******************************* colonoscopy image with adenomatous polyp, white light imaging, good-quality, clear ***************************************************#
images = pipe(prompt="colonoscopy image with adenomatous polyp, white light imaging, good-quality, clear", negative_prompt= n_prompt, num_images_per_prompt=30).images
for i, image in enumerate(images):
    image.save("./output_image_dir/polyp/WL/AD/"+str(i)+".jpg")    
 
#Section 10:   
#******************************* colonoscopy image with hyperplastic polyp, white light imaging, good-quality, clear ***************************************************#    
images = pipe(prompt="colonoscopy image with hyperplastic polyp, white light imaging, good-quality, clear", negative_prompt= n_prompt, num_images_per_prompt=30).images
for i, image in enumerate(images):
    image.save("./output_image_dir/polyp/WL/HP/"+str(i)+".jpg") 

#Section 11:
#******************************* colonoscopy image with adenomatous polyp, narrow band imaging, good-quality, clear ***************************************************#     
images = pipe(prompt="colonoscopy image with adenomatous polyp, narrow band imaging, good-quality, clear", negative_prompt= n_prompt, num_images_per_prompt=30).images
for i, image in enumerate(images):
    image.save("./output_image_dir/polyp/NBI/AD/"+str(i)+".jpg")    
    
#Section 12:    
#******************************* colonoscopy image with hyperplastic polyp, narrow band imaging, good-quality, clear ***************************************************#     
images = pipe(prompt="colonoscopy image with hyperplastic polyp, narrow band imaging, good-quality, clear", negative_prompt= n_prompt, num_images_per_prompt=30).images
for i, image in enumerate(images):
    image.save("./output_image_dir/polyp/NBI/HP/"+str(i)+".jpg") 
    
    
    
    
#Section 13:    
#**************************** With Weighted Control Mechanism (colonoscopy image with hyperplastic polyp, narrow band imaging, good-quality, clear) ****************************# 
compel_proc = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)
images = pipe(prompt_embeds=compel_proc("(colonoscopy image with hyperplastic polyp, narrow band imaging)++, good-quality, clear"), negative_prompt= n_prompt, num_images_per_prompt=30).images
for i, image in enumerate(images):
    image.save("./output_image_dir/polyp/NBI/HP/"+str(i)+".jpg")

#Section 14:    
#**************************** With Weighted Control Mechanism (colonoscopy image with adenomatous polyp, narrow band imaging, good-quality, clear) ****************************#   
compel_proc = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)          
images = pipe(prompt_embeds=compel_proc("(colonoscopy image with adenomatous polyp, narrow band imaging)++, good-quality, clear"), negative_prompt= n_prompt, num_images_per_prompt=30).images
for i, image in enumerate(images):
    image.save("./output_image_dir/polyp/NBI/AD/"+str(i)+".jpg")
'''
    

        
        

