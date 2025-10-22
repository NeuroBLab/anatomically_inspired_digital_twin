import numpy as np
import os
from tqdm import tqdm

def main():
    # Set random seed
    np.random.seed(42)

    # Generate 500,000 arrays of white noise
    num_samples = 500_000
    image_shape = (36, 64)
    images = np.random.uniform(0, 255, size=(num_samples, *image_shape))

    # Generate metadata arrays
    trial_idx = np.arange(num_samples)
    tiers = np.array(["test"] * num_samples, dtype='<U4')
    seed = np.array([42], dtype=int)

    # Define folder paths
    base_path = "data/white_noise/seed"

    data_path = os.path.join(base_path, "data")
    videos_path = os.path.join(data_path, "videos")
    pupil_center_path = os.path.join(data_path, "pupil_center")
    responses_path = os.path.join(data_path, "responses")
    behavior_path = os.path.join(data_path, "behavior")

    meta_path = os.path.join(base_path, "meta")
    trials_path = os.path.join(meta_path, "trials")
    statistics_path = os.path.join(meta_path, "statistics")
    neurons_path = os.path.join(meta_path, "neurons")

    # Create directories
    for path in [
        base_path, 
        data_path, 
        meta_path, 
        videos_path,
        pupil_center_path, 
        responses_path, 
        behavior_path,
        trials_path,
        statistics_path,
        neurons_path
    ]:
        os.makedirs(path, exist_ok=True)

    behavior_and_pupil = np.zeros((2, 1))
    responses = np.zeros((1, 1))
    # Save images as individual .npy files
    for i, img in enumerate(tqdm(images, desc="Saving images")):
        np.save(os.path.join(videos_path, f"{i}.npy"), img)
        np.save(os.path.join(pupil_center_path, f"{i}.npy"), behavior_and_pupil)
        np.save(os.path.join(responses_path, f"{i}.npy"), responses)
        np.save(os.path.join(behavior_path, f"{i}.npy"), behavior_and_pupil)
        
    # Save metadata as individual .npy files
    np.save(os.path.join(trials_path, "trial_idx.npy"), trial_idx)
    np.save(os.path.join(trials_path, "tiers.npy"), tiers)
    np.save(os.path.join(trials_path, "seed.npy"), seed)

    print("All images saved successfully!")
    
if __name__ == "__main__":
    main()