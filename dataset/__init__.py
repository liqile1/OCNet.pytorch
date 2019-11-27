from .cityscapes import CitySegmentationTrain, CitySegmentationTest, CitySegmentationTrainWpath
from .leadbang import LeadBangTest, LeadBangTrain

datasets = {
	'cityscapes_train': CitySegmentationTrain,
	'cityscapes_test': CitySegmentationTest,
	'cityscapes_train_w_path': CitySegmentationTrainWpath,
	'leadbang_train': LeadBangTrain,
	'leadbang_test': LeadBangTest,
}


def get_segmentation_dataset(name, **kwargs):
    return datasets[name.lower()](**kwargs)