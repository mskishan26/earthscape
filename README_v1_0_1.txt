This README.txt file was updated on 07-16-2025 by Matthew A. Massey, Kentucky Geological Survey



GENERAL INFORMATION

1. Title of Dataset: 
- EarthScape AI Dataset (VERSION 1.0.1)


2. Contact Information
- Name: Matthew A. Massey
Institution: Kentucky Geological Survey
Address: University of Kentucky, Lexington, KY, 40506-0107
Email: matthew.massey@uky.edu	

- Name: Abdullah-Al-Zubaer Imran
Institution: Department of Computer Science, University of Kentucky
Address: University of Kentucky, Lexington, KY, 40506-0633
Email: aimran@uky.edu	


3. Date of data collection: 
- NA


4. Geographic location of data collection: 
- This dataset currently includes image patches and labels from ten 7.5-minute quadrangles in and around Warren and Hardin Counties, Kentucky. 


5. Information about funding sources that supported the collection of the data: 
- Funding for the original mapping has been provided by the Kentucky Geological Survey.


6. Abstract:
- Surficial geologic mapping is essential for understanding Earth surface processes, addressing modern challenges such as climate change and national security, and supporting common applications in engineering and resource management. However, traditional mapping methods are labor-intensive, limiting spatial coverage and introducing potential biases. To address these limitations, we introduce EarthScape, a novel, AI-ready multimodal dataset specifically designed for surficial geologic mapping and Earth surface analysis. EarthScape integrates high-resolution aerial RGB and near-infrared (NIR) imagery, digital elevation models (DEM), multi-scale DEM-derived terrain features, and hydrologic and infrastructure vector data. The dataset provides masks and labels for seven surficial geologic classes encompassing various geological processes. As a living dataset with a vision for expansion, EarthScape bridges the gap between computer vision and Earth sciences, offering a valuable resource for advancing research in multimodal learning, geospatial analysis, and geological mapping. The reader is referred to the current arXiv manuscript (https://arxiv.org/abs/2503.15625) and GitHub repository (https://github.com/masseygeo/earthscape) for additional details.





SHARING/ACCESS INFORMATION

1. Licenses/restrictions placed on the data:
- © 2025 University of Kentucky
This dataset is distributed under the terms of the Creative Commons Attribution 4.0 International License (https://creativecommons.org/licenses/by/4.0/), which permits unrestricted use, distribution, and reproduction in any medium, provided that the dataset creators and source are credited and that changes (if any) are clearly indicated.


2. Links to publications that cite or use the data: 
- https://arxiv.org/abs/2503.15625


3. Links to other publicly accessible locations of the data: 
- https://github.com/masseygeo/earthscape


4. Links/relationships to ancillary data sets:
- NA


5. Was data derived from another source? 
- Kentucky Geological surficial geologic maps
	- https://uknowledge.uky.edu/kgs_data/
- KyFromAbove Program
	- https://kyfromabove.ky.gov/
- National Hydrography Dataset
	- https://www.usgs.gov/national-hydrography/national-hydrography-dataset
- OpenStreetMaps
	- https://www.openstreetmap.org/about


6. Recommended citation for this dataset: 
- Massey, M.A., and Imran, A., 2025, EarthScape AI Dataset [ver. 1.0.0]: Kentucky Geological Survey, ser. 14, Research Data, DOI: XXX.  




DATA & FILE OVERVIEW

1. ZIP File Extraction
- EarthScape is packaged as a large ZIP archives, and we have observed frequent corruption or incomplete extraction when using the default "right-click" or Finder-based extraction tools on Windows and macOS. To ensure the dataset is properly extracted, we strongly recommend using a terminal-based extraction method as shown below:

Windows (Powershell)
Expand-Archive -Path "C:\Path\To\Your\Dataset.zip" -DestinationPath "C:\Path\To\Extract\Folder" -Force

macOS (Terminal)
unzip /path/to/your/dataset.zip -d /path/to/extract/folder


2. File List: 
- README_v1_0_1.txt - This text document.

- DataDictionary_v1_0_1.txt - Data dictionary.

- earthscape_EXAMPLE_EXAMPLE.zip - ZIP file containing small example of EarthScape data for inspection purposes.

- earthscape_warren_bgn.zip - ZIP files containing EarthScape dataset files for the Bowling Green North 7.5-minute quadrangle.

- earthscape_warren_bgs.zip - ZIP files containing EarthScape dataset files for the Bowling Green South 7.5-minute quadrangle.

- earthscape_warren_bristow.zip - ZIP files containing EarthScape dataset files for the Bristow 7.5-minute quadrangle.

- earthscape_warren_hadley.zip - ZIP files containing EarthScape dataset files for the Hadley 7.5-minute quadrangle.

- earthscape_warren_rockfield.zip - ZIP files containing EarthScape dataset files for the Rockfield 7.5-minute quadrangle.

- earthscape_warren_smithsgrove.zip - ZIP files containing EarthScape dataset files for the Smiths Grove 7.5-minute quadrangle.

- earthscape_hardin_howevalley.zip - ZIP files containing EarthScape dataset files for the Howe Valley 7.5-minute quadrangle.

- earthscape_hardin_sonora.zip - ZIP files containing EarthScape dataset files for the Sonora 7.5-minute quadrangle.

* All ZIP files can be described with the following structure:
- earthscape_{subset_area}_{quadrangle}/patches - Directory containing GeoTIFF files for each patch location in relevant area; files named by unique patch ID and modality (e.g., {patch_id}_{modality}.tif). Each patch location is associated with 38 images, including surficial geologic mask, DEM, RGB+NIR, NHD, OSM, and terrain features (slope, profile curvature, planform curvature, standard deviation of slope, elevation percentile terrain features calculated at six spatial scales); each patch location is also associated with one .csv file containing one-hot encoded labels for the patch.

- earthscape_{subset_area}_{quadrangle}/locations.geojson - Vector GIS file with patch location footprints.

- earthscape_{subset_area}_{quadrangle}/labels.csv - Table of one-hot encoded labels for each of the seven geologic classes at each of the patch locations.

- earthscape_{subset_area}_{quadrangle}/areas.csv - Table of class proportions present in each patch.


3. Relationship between files, if important: 
- patch_id is a common field name in earthscape_{subset_area}_{quadrangle}/locations.geojson, earthscape_{subset_area}_{quadrangle}/labels.csv, and earthscape_{subset_area}_{quadrangle}/areas.csv that allows primary-foreign key relationships. GeoTIFF files in earthscape_{subset_area}_{quadrangle}/patches are named in a {patch_id}_{modality}.tif format that allows linking other GeoJSON and CSV files.  


4. Additional related data collected that was not included in the current data package: 
- N/A


5. Are there multiple versions of the dataset?
- This is a versioned dataset, and the current version is v1.0.0. New map (masks) and feature data will be added as new maps become available, and existing maps products are verified for quality control. Dataset updates will be released as a new version, and will be summarized in this README document.




METHODOLOGICAL INFORMATION

1. Description of methods used for collection/generation of data:
- The reader is referred to the following open-access sources for more details about the dataset:
	- https://arxiv.org/abs/2503.15625
	- https://github.com/masseygeo/earthscape


2. Methods for processing the data:
- The reader is referred to the following open-access sources for more details about the dataset:
	- https://arxiv.org/abs/2503.15625
	- https://github.com/masseygeo/earthscape


3. Instrument- or software-specific information needed to interpret the data: 
- The data has been pre-processed and ready for AI modeling. The GeoTIFF format allows for inspection with any GIS software. In Python, the Rasterio library provides easy open-access tools for visualization:

import rasterio
from rasterio.plot import show
with rasterio.open("GEOTIFF FILE PATH") as src:
     show(src)


4. Standards and calibration information, if appropriate: 
- NA


5. Environmental/experimental conditions: 
- NA


6. Describe any quality-assurance procedures performed on the data: 
- The target surficial geologic map GeoTIFF images served as the spatial reference for aligning all other features in the dataset. Once each feature was collected and compiled into its respective GeoTIFF image format, they were each reprojected to align with the reference image coordinates using cubic convolution interpolation. All images were checked to ensure that their bounding coordinates and spatial resolutions were identical across all other images.

- Vector polygon patches were systematically constructed in a grid pattern to cover the target AOIs using the same coordinate reference system, each polygon was assigned a unique patch ID, and then all patches were saved as a GeoJSON file. Each polygon patch was constructed so that it covers an area of exactly 1280 × 1280 feet (256 × 256 pixels), overlaps adjacent patches by 50%, and is fully contained within the target AOI. The vector patch GeoJSON was then used to extract 38 channels for each patch, including target mask, aerial RGB and NIR, DEM, the five terrain features calculated at six scales, NHD, and OSM. The vector patch GeoJSON was also spatially joined with the target surficial geologic map GeoJSON to calculate both one-hot encoded labels and the proportional areas occupied by each target class within each patch, available as a .csv file that can be linked to each patch using the unique patch IDs.


7. People involved with sample collection, processing, analysis and/or submission: 
- This compilation has been processed and assembled by Matthew A. Massey and Abdullah-Al-Zubaer Imran.

