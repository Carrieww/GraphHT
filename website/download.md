---
layout: page
title: Download
permalink: /download/
---

# Download

## Code

The latest version of the code is available on [GitHub](https://github.com/Carrieww/GraphHT).

### Requirements

- Python 3.x
- Required packages (see `requirements.txt`):
  ```
  numpy
  pandas
  networkx
  scipy
  ```

## Datasets

Due to file size limitations on GitHub, the datasets are available for download from external sources:

1. **MovieLens Dataset**
   - Description: Movie rating dataset with user-movie interactions
   - Size: ~100MB
   - [Download Link](#) (Please contact us for access)

2. **DBLP Dataset**
   - Description: Academic collaboration network
   - Size: ~500MB
   - [Download Link](#) (Please contact us for access)

3. **Yelp Dataset**
   - Description: Business review network
   - Size: ~600MB
   - [Download Link](#) (Please contact us for access)

After downloading the datasets, place them in the `datasets/` directory of your local repository.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Carrieww/GraphHT.git
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Download and place the datasets in the `datasets/` directory

4. Run the framework:
   ```bash
   python main.py --sampling_method "PHASE"
   ```
   or
   ```bash
   bash run.sh
   ```

## Documentation

For detailed documentation, please refer to the [GitHub repository](https://github.com/Carrieww/GraphHT). 