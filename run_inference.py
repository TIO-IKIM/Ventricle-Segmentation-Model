"""
Authors: Lukas Heine, Fabian Hörst, Jana Fragemann, Moritz Rempe, Frederic Jonske, Gijs Luijten
Institute for Artificial Intelligence in Medicine
University Hospital Essen
"""
import os
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import torch
import tempfile
import logging
from rt_utils import RTStructBuilder
import argparse
import nibabel as nib
import numpy as np
import glob
import SimpleITK as sitk

def converter(
        dicom_series_path: str,
        multilabel_mask_path: str,
        name: str,
        save_path: str
) -> str:
    """Construct an RTSTRUCT file from a given dicom series and a multilabel mask"""
    multilabel_mask = nib.load(multilabel_mask_path).get_fdata()
    # rotate mask
    multilabel_mask = np.fliplr(np.rot90(multilabel_mask, k=-1))
    rtstruct = RTStructBuilder.create_new(dicom_series_path=dicom_series_path)
    # make binary ventricle and brain stem mask from multilabel mask
    # ventricle label 1
    label1_mask = np.array(multilabel_mask == 1, dtype=bool)
    # brain stem label 2
    label2_mask = np.array(multilabel_mask == 2, dtype=bool)
    # mask for ventricle
    rtstruct.add_roi(mask=label1_mask, color=[0, 0, 255], name="Ventricle")
    # mask for brain stem
    rtstruct.add_roi(mask=label2_mask, color=[0, 255, 255], name="Brain stem")

    file_handle = os.path.join(save_path, f"prediction_{name}.dcm")
    rtstruct.save(file_handle)
    return file_handle


def dcm2nifti(dir_path: str, out_path: str) -> None:
    """
    Load dicom slices and store them in a volume
    :param dir_path: Directory where the dicom files are stored
    :param out_path: Store path of the Nifti file
    :return: None
    """
    reader = sitk.ImageSeriesReader()
    dcm_series = reader.GetGDCMSeriesFileNames(dir_path)
    reader.SetFileNames(dcm_series)
    image = reader.Execute()

    sitk.WriteImage(image, out_path)

class arguments():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
    def get_folder_path(self)->object:
        self.parser.add_argument('--input_folder',help='.')
        self.parser.add_argument('--output_folder',help='.')
        self.parser.add_argument('--mr_sequence',help='T1 or T2')
        return self.parser

class VenSegModel():
    def __init__(self) -> None:
        """
        :param name: Name of the model instance
        """
        self.predictor = None
        self.folder = None
        self.args = arguments().get_folder_path().parse_args()

        match self.args.mr_sequence.lower():
            case 't1':
                self.folder = 'Dataset501_VenSeg3DT1'
            case 't2':
                self.folder = 'Dataset502_VenSeg3DT2'
            case _:
                raise TypeError(f'Unknown sequence {self.args.mr_sequence}')

        # Set nnunet variables
        os.environ['nnUNet_results'] = os.path.sep.join(__file__.split(os.path.sep)[:-1] + ['nnUNet_results'])
        self.load()
    
    def load(self) -> None:
        assert torch.cuda.is_available(), f'No valid CUDA GPU found, please check your container and run command'
        logging.info('Using CUDA GPU')
        self.predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device=True,
            device=torch.device('cuda', 0),
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=False
        )
        # initializes the network architecture, loads the checkpoint
        self.predictor.initialize_from_trained_model_folder(
            os.path.join(os.environ['nnUNet_results'], self.folder, 'nnUNetTrainer__nnUNetPlans__2d'),
            use_folds=None,
            checkpoint_name='checkpoint_best.pth')

        logging.info('Model ready!')
       
    def predict(self) -> None:
        if glob.glob(os.path.join(self.args.input_folder,'*.dcm',case_sensitive=False)):
            with tempfile.TemporaryDirectory(dir='.') as in_dir:
            
                name = self.args.input_folder.split(os.path.sep)[-1] +f'_0000.nii.gz'
            
                input_file = os.path.join(in_dir, name)
                
                #convert files to nifti
                dcm2nifti(self.args.input_folder, input_file)
                logging.info(f'Saved Nifti to {input_file}')
                with tempfile.TemporaryDirectory(dir='.') as out_dir:
                    self.predictor.predict_from_files(
                                                        list_of_lists_or_source_folder=in_dir,
                                                        output_folder_or_list_of_truncated_output_files=out_dir
                                                        )
                    name = self.args.input_folder.split(os.path.sep)[-1] +'.nii.gz'
                    nifti_mask = os.path.join(out_dir, name)
                    converter(
                                dicom_series_path=self.args.input_folder,
                                multilabel_mask_path=nifti_mask,
                                name=name.removesuffix('.nii.gz'),
                                save_path=self.args.output_folder
                             )
        
        elif os.listdir(self.args.input_folder)[0].endswith('nii.gz'):
            self.predictor.predict_from_files(
                                                list_of_lists_or_source_folder=self.args.input_folder,
                                                output_folder_or_list_of_truncated_output_files=self.args.output_folder
                                                )
        else:
            raise TypeError(f'Unknown dataformat.')
                           
if __name__ == "__main__":
    model = VenSegModel()
    model.predict()
