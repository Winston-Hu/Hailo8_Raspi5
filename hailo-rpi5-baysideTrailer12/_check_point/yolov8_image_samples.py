import degirum as dg, degirum_tools

inference_host_address = "@local"
zoo_url = 'degirum/hailo'
token=''
device_type=['HAILORT/HAILO8L']

# choose a model to run inference on by uncommenting one of the following lines
model_name = "yolov8n_coco--640x640_quant_hailort_multidevice_1"
# model_name = "yolov8n_relu6_coco_pose--640x640_quant_hailort_hailo8l_1"
# model_name = "yolov8n_relu6_coco_seg--640x640_quant_hailort_hailo8l_1"
# model_name = "yolov8s_silu_imagenet--224x224_quant_hailort_hailo8l_1"

# choose image source
image_source = "../assets/ThreePersons.jpg"

# load AI model
model = dg.load_model(
    model_name=model_name,
    inference_host_address=inference_host_address,
    zoo_url=zoo_url,
    token=token,
    device_type=device_type
)

# perform AI model inference on given image source
print(f" Running inference using '{model_name}' on image source '{image_source}'")
inference_result = model(image_source)

# print('Inference Results \n', inference_result)  # numeric results
print(inference_result)
print("Press 'x' or 'q' to stop.")

# show results of inference
with degirum_tools.Display("AI Camera") as output_display:
    output_display.show_image(inference_result.image_overlay)