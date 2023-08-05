from nnunet.dataset_conversion.utils import generate_dataset_json

if __name__ == '__main__':
    
    json_path = r''
    imagesTr_dir = r''
    imagesTs_dir = r''
    generate_dataset_json(json_path,
                          imagesTr_dir = imagesTr_dir, 
                          imagesTs_dir = imagesTs_dir, 
                          modalities = ['MRA'],
                          labels={0: 'background', 1: 'CereVess'}, 
                          dataset_name='Task501_CereVess', 
                          license='!')