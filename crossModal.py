import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os
import librosa
import cv2
import pandas as pd
from tqdm import tqdm
import soundfile as sf  # For writing WAV files
import mediapipe as mp

# Define the cross-modal architecture
class CrossModalHCI(nn.Module):
    def __init__(self, voice_input_dim, gesture_input_dim, eye_input_dim, shared_dim, output_dim):
        super(CrossModalHCI, self).__init__()
        
        # Voice branch
        self.voice_encoder = nn.Sequential(
            nn.Linear(voice_input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, shared_dim)
        )
        
        # Gesture branch
        self.gesture_encoder = nn.Sequential(
            nn.Linear(gesture_input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, shared_dim)
        )
        
        # Eye tracking branch
        self.eye_encoder = nn.Sequential(
            nn.Linear(eye_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, shared_dim)
        )
        
        # Fusion mechanism
        self.attention = nn.MultiheadAttention(embed_dim=shared_dim, num_heads=4)
        
        # Shared representation layers
        self.shared_layers = nn.Sequential(
            nn.Linear(shared_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Output layer
        self.output_layer = nn.Linear(64, output_dim)
        
    def forward(self, voice_input, gesture_input, eye_input):
        # Encode each modality
        voice_features = self.voice_encoder(voice_input)
        gesture_features = self.gesture_encoder(gesture_input)
        eye_features = self.eye_encoder(eye_input)
        
        # Stack features for attention mechanism
        features = torch.stack([voice_features, gesture_features, eye_features], dim=0)
        
        # Apply self-attention to fuse modalities
        attended_features, _ = self.attention(features, features, features)
        
        # Average the attended features
        fused_features = torch.mean(attended_features, dim=0)
        
        # Pass through shared layers
        shared_representation = self.shared_layers(fused_features)
        
        # Output layer
        output = self.output_layer(shared_representation)
        
        return output, voice_features, gesture_features, eye_features, shared_representation

# Custom dataset for cross-modal HCI data
class HCIDataset(Dataset):
    def __init__(self, voice_features, gesture_features, eye_features, labels):
        self.voice_features = voice_features
        self.gesture_features = gesture_features
        self.eye_features = eye_features
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'voice': self.voice_features[idx],
            'gesture': self.gesture_features[idx],
            'eye': self.eye_features[idx],
            'label': self.labels[idx]
        }

def extract_eye_features_from_image(image_path):
    """Extract features from eye regions in an image using MediaPipe"""
    try:
        # Initialize MediaPipe Face Mesh
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5)
        
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not read image: {image_path}")
            return np.zeros(50)  # Return zeros if image can't be read
        
        # Convert to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape
        
        # Process the image
        results = face_mesh.process(image_rgb)
        
        if not results.multi_face_landmarks:
            print(f"No face detected in {image_path}")
            return np.zeros(50)
        
        face_landmarks = results.multi_face_landmarks[0]
        
        # Eye landmarks indices (MediaPipe Face Mesh)
        # Left eye indices: 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398
        # Right eye indices: 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246
        
        left_eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        right_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        
        # Extract landmark coordinates
        landmarks = np.array([(lm.x * w, lm.y * h) for lm in face_landmarks.landmark])
        
        # Get eye landmarks
        left_eye_points = landmarks[left_eye_indices]
        right_eye_points = landmarks[right_eye_indices]
        
        # Calculate eye features
        features = []
        
        # Left eye bounding box
        left_min_x, left_min_y = np.min(left_eye_points, axis=0)
        left_max_x, left_max_y = np.max(left_eye_points, axis=0)
        left_width = left_max_x - left_min_x
        left_height = left_max_y - left_min_y
        left_aspect_ratio = left_width / left_height if left_height > 0 else 0
        
        # Right eye bounding box
        right_min_x, right_min_y = np.min(right_eye_points, axis=0)
        right_max_x, right_max_y = np.max(right_eye_points, axis=0)
        right_width = right_max_x - right_min_x
        right_height = right_max_y - right_min_y
        right_aspect_ratio = right_width / right_height if right_height > 0 else 0
        
        # Add centroids
        left_center = np.mean(left_eye_points, axis=0)
        right_center = np.mean(right_eye_points, axis=0)
        
        # Distance between eyes
        eye_distance = np.linalg.norm(left_center - right_center)
        
        # Add features
        features.extend([
            # Left eye features
            left_min_x, left_min_y, left_max_x, left_max_y,
            left_width, left_height, left_aspect_ratio,
            left_center[0], left_center[1],
            
            # Right eye features
            right_min_x, right_min_y, right_max_x, right_max_y,
            right_width, right_height, right_aspect_ratio,
            right_center[0], right_center[1],
            
            # Inter-eye features
            eye_distance,
        ])
        
        # Calculate the average pixel values in eye regions (simple proxy for pupil size/openness)
        left_eye_roi = image[int(left_min_y):int(left_max_y), int(left_min_x):int(left_max_x)]
        right_eye_roi = image[int(right_min_y):int(right_max_y), int(right_min_x):int(right_max_x)]
        
        if left_eye_roi.size > 0:
            left_eye_gray = cv2.cvtColor(left_eye_roi, cv2.COLOR_BGR2GRAY)
            left_eye_avg = np.mean(left_eye_gray)
            left_eye_std = np.std(left_eye_gray)
        else:
            left_eye_avg, left_eye_std = 0, 0
            
        if right_eye_roi.size > 0:
            right_eye_gray = cv2.cvtColor(right_eye_roi, cv2.COLOR_BGR2GRAY)
            right_eye_avg = np.mean(right_eye_gray)
            right_eye_std = np.std(right_eye_gray)
        else:
            right_eye_avg, right_eye_std = 0, 0
        
        features.extend([
            left_eye_avg, left_eye_std,
            right_eye_avg, right_eye_std
        ])
        
        # Get gaze direction (approximation using iris position relative to eye corners)
        # This is a very simplified approach
        def get_gaze_ratio(eye_points, landmarks):
            # Get eye corner points
            corner_left = landmarks[eye_points[0]]
            corner_right = landmarks[eye_points[8]]
            
            # Get the center of the eye
            eye_center = np.mean(landmarks[eye_points], axis=0)
            
            # Calculate distances
            dist_to_left = np.linalg.norm(eye_center - corner_left)
            dist_to_right = np.linalg.norm(eye_center - corner_right)
            
            # Calculate gaze ratio
            if dist_to_right > 0:
                return dist_to_left / dist_to_right
            return 1.0
        
        left_gaze_ratio = get_gaze_ratio(left_eye_indices, landmarks)
        right_gaze_ratio = get_gaze_ratio(right_eye_indices, landmarks)
        
        features.extend([
            left_gaze_ratio,
            right_gaze_ratio
        ])
        
        # Pad if we don't have enough features
        while len(features) < 50:
            features.append(0)
        
        face_mesh.close()
        
        return np.array(features[:50])  # Take only the first 50 features
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return np.zeros(50)

# Feature extraction functions
def extract_voice_features(audio_file, max_length=128):
    """Extract MFCC features from voice audio"""
    try:
        y, sr = librosa.load(audio_file, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfccs = mfccs.T  # Transpose to get time steps as first dimension
        
        # Pad or truncate to max_length
        if mfccs.shape[0] < max_length:
            pad_width = max_length - mfccs.shape[0]
            mfccs = np.pad(mfccs, ((0, pad_width), (0, 0)), mode='constant')
        else:
            mfccs = mfccs[:max_length, :]
        
        return mfccs.flatten()
    except Exception as e:
        print(f"Error processing {audio_file}: {e}")
        return np.zeros(40 * max_length)

def extract_gesture_features(video_file, num_frames=32):
    """Extract pose features from gesture video using OpenPose or MediaPipe"""
    try:
        cap = cv2.VideoCapture(video_file)
        frames = []
        
        while len(frames) < num_frames and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Here you would typically use a pose estimation library
            # For simplicity, we'll just resize the frame and convert to grayscale
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.resize(frame, (64, 64))
            frames.append(frame.flatten())

        cap.release()
        
        # Pad if we don't have enough frames
        while len(frames) < num_frames:
            frames.append(np.zeros_like(frames[0]))
        
        # Concatenate all frames
        return np.concatenate(frames[:num_frames])
        
    except Exception as e:
        print(f"Error processing {video_file}: {e}")
        return np.zeros(64 * 64 * num_frames)

def extract_eye_features(eye_data_file):
    """Extract features from eye tracking data file"""
    try:
        # Assuming eye_data_file is a CSV with columns like timestamp, x, y, pupil_size
        df = pd.read_csv(eye_data_file)
        
        # Extract statistical features
        features = []
        
        # Position statistics
        for col in ['x', 'y']:
            if col in df.columns:
                features.extend([
                    df[col].mean(),
                    df[col].std(),
                    df[col].min(),
                    df[col].max()
                ])
        
        # Pupil size statistics
        if 'pupil_size' in df.columns:
            features.extend([
                df['pupil_size'].mean(),
                df['pupil_size'].std(),
                df['pupil_size'].min(),
                df['pupil_size'].max()
            ])
        
        # Fixation duration
        if 'fixation_duration' in df.columns:
            features.extend([
                df['fixation_duration'].mean(),
                df['fixation_duration'].std(),
                df['fixation_duration'].sum()
            ])
        
        # Saccade features
        if 'saccade_length' in df.columns:
            features.extend([
                df['saccade_length'].mean(),
                df['saccade_length'].std(),
                df['saccade_length'].sum()
            ])
        
        # If we don't have enough features, pad with zeros
        while len(features) < 50:  # Arbitrary feature length
            features.append(0)
        
        return np.array(features[:50])  # Take only the first 50 features
        
    except Exception as e:
        print(f"Error processing {eye_data_file}: {e}")
        return np.zeros(50)

# Generate synthetic data
def generate_synthetic_data(num_samples=100, num_classes=5, data_dir='synthetic_data'):
    # Create directories
    os.makedirs(os.path.join(data_dir, 'voice'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'gesture'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'eye'), exist_ok=True)
    
    labels = []
    
    for i in tqdm(range(num_samples), desc="Generating synthetic data"):
        participant_id = f"P{i//10 + 1:03d}"
        task_id = f"T{i%10 + 1:03d}"
        label = np.random.randint(0, num_classes)
        
        # Generate synthetic voice file (white noise with some structure)
        sr = 16000
        duration = 2.0  # seconds
        t = np.linspace(0, duration, int(sr * duration))
        # Create a mix of sine waves with noise
        freq = np.random.uniform(80, 400)  # Random base frequency
        audio = 0.5 * np.sin(2 * np.pi * freq * t)
        audio += 0.3 * np.sin(2 * np.pi * freq * 2 * t)  # Harmonic
        audio += 0.1 * np.random.randn(len(t))  # Add noise
        audio = audio / np.max(np.abs(audio))  # Normalize
        
        voice_file = os.path.join(data_dir, 'voice', f"{participant_id}_{task_id}.wav")
        sf.write(voice_file, audio, sr)
        
        # Generate synthetic gesture file (moving patterns rather than random frames)
        video_file = os.path.join(data_dir, 'gesture', f"{participant_id}_{task_id}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_file, fourcc, 30, (64, 64), False)
        
        # Create a moving pattern
        for frame_idx in range(30):  # 30 frames
            # Create a moving circle or pattern
            frame = np.zeros((64, 64), dtype=np.uint8)
            center_x = int(32 + 20 * np.sin(frame_idx / 30 * 2 * np.pi))
            center_y = int(32 + 15 * np.cos(frame_idx / 30 * 2 * np.pi))
            cv2.circle(frame, (center_x, center_y), 5, 255, -1)
            out.write(frame)
        
        out.release()
        
        # Generate synthetic eye tracking data with realistic patterns
        eye_file = os.path.join(data_dir, 'eye', f"{participant_id}_{task_id}.csv")
        
        num_points = 200
        timestamps = np.linspace(0, 2, num_points)
        
        # Generate more realistic eye movements (similar to reading or scanning patterns)
        t = np.linspace(0, 2*np.pi, num_points)
        # Base position with some randomness
        x_center = 960 + np.random.normal(0, 100)
        y_center = 540 + np.random.normal(0, 80)
        # Movement amplitudes
        x_amp = np.random.uniform(100, 400)
        y_amp = np.random.uniform(50, 200)
        # Frequencies
        x_freq = np.random.uniform(0.5, 2.5)
        y_freq = np.random.uniform(0.3, 1.5)
        
        # Generate coordinates with some randomness
        x = x_center + x_amp * np.sin(t * x_freq) + np.random.normal(0, 20, num_points)
        y = y_center + y_amp * np.sin(t * y_freq + np.pi/3) + np.random.normal(0, 15, num_points)
        
        # Clip to screen bounds
        x = np.clip(x, 0, 1920)
        y = np.clip(y, 0, 1080)
        
        # Generate pupil size with realistic variations
        base_pupil = np.random.uniform(2.5, 4.0)
        pupil_size = base_pupil + 0.5 * np.sin(t * 0.5) + np.random.normal(0, 0.1, num_points)
        
        # Generate fixation durations
        fixation_mask = np.random.rand(num_points) > 0.2  # 80% chance of being in a fixation
        fixation_duration = np.zeros(num_points)
        current_fixation = 0
        
        for i in range(num_points):
            if fixation_mask[i]:
                current_fixation += timestamps[1] - timestamps[0]  # Increment by time step
            else:
                current_fixation = 0  # Reset during saccades
            fixation_duration[i] = current_fixation
        
        # Generate saccade lengths
        saccade_length = np.zeros(num_points)
        for i in range(1, num_points):
            if not fixation_mask[i]:
                # Calculate distance moved during saccade
                saccade_length[i] = np.sqrt((x[i] - x[i-1])**2 + (y[i] - y[i-1])**2)
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'x': x,
            'y': y,
            'pupil_size': pupil_size,
            'fixation_duration': fixation_duration,
            'saccade_length': saccade_length
        })
        df.to_csv(eye_file, index=False)
        
        labels.append({
            'participant_id': participant_id,
            'task_id': task_id,
            'label': label
        })
    
    # Create labels CSV
    labels_df = pd.DataFrame(labels)
    labels_df.to_csv(os.path.join(data_dir, 'labels.csv'), index=False)
    
    print(f"Generated synthetic dataset with {num_samples} samples in {data_dir}")
    return os.path.join(data_dir, 'labels.csv')

# Function to prepare dataset
def prepare_dataset(data_dir, label_file, eye_images_dir):
    labels_df = pd.read_csv(label_file)
    
    voice_features_list = []
    gesture_features_list = []
    eye_features_list = []
    labels_list = []
    
    for _, row in tqdm(labels_df.iterrows(), total=len(labels_df), desc="Extracting features"):
        participant_id = row['participant_id']
        task_id = row['task_id']
        label = row['label']-1
        
        # Construct file paths
        voice_file = os.path.join(data_dir, 'voice', f"{participant_id}_{task_id}.wav")
        gesture_file = os.path.join(data_dir, 'gesture', f"{participant_id}_{task_id}.mp4")
        eye_image_file = os.path.join(eye_images_dir, f"{participant_id}_{task_id}.jpg")  # Assuming jpg format
        
        # Check for other possible extensions if jpg doesn't exist
        if not os.path.exists(eye_image_file):
            for ext in ['.png', '.jpeg', '.bmp']:
                alt_path = os.path.join(eye_images_dir, f"{participant_id}_{task_id}{ext}")
                if os.path.exists(alt_path):
                    eye_image_file = alt_path
                    break
        
        # Check if all files exist
        if not os.path.exists(voice_file):
            print(f"Missing voice file for participant {participant_id}, task {task_id}")
            continue
        if not os.path.exists(gesture_file):
            print(f"Missing gesture file for participant {participant_id}, task {task_id}")
            continue
        if not os.path.exists(eye_image_file):
            print(f"Missing eye image file for participant {participant_id}, task {task_id}")
            continue
        
        # Extract features
        voice_features = extract_voice_features(voice_file)
        gesture_features = extract_gesture_features(gesture_file)
        eye_features = extract_eye_features_from_image(eye_image_file)
        
        # Append to lists
        voice_features_list.append(voice_features)
        gesture_features_list.append(gesture_features)
        eye_features_list.append(eye_features)
        labels_list.append(label)
    
    # Convert to numpy arrays
    voice_features_array = np.array(voice_features_list)
    gesture_features_array = np.array(gesture_features_list)
    eye_features_array = np.array(eye_features_list)
    labels_array = np.array(labels_list)
    
    return voice_features_array, gesture_features_array, eye_features_array, labels_array

# Training function with transfer learning
def train_model(model, train_loader, val_loader, num_epochs=30, learning_rate=0.001,
               lambda_transfer=0.1, device='cuda'):
    
    # Move model to device
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            voice_input = batch['voice'].float().to(device)
            gesture_input = batch['gesture'].float().to(device)
            eye_input = batch['eye'].float().to(device)
            labels = batch['label'].long().to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs, voice_feat, gesture_feat, eye_feat, shared_rep = model(voice_input, gesture_input, eye_input)
            
            # Task loss
            task_loss = criterion(outputs, labels)
            
            # Transfer loss - encourage shared representations across modalities
            v_g_transfer = nn.functional.mse_loss(voice_feat, gesture_feat)
            v_e_transfer = nn.functional.mse_loss(voice_feat, eye_feat)
            g_e_transfer = nn.functional.mse_loss(gesture_feat, eye_feat)
            
            transfer_loss = v_g_transfer + v_e_transfer + g_e_transfer
            
            # Total loss
            loss = task_loss + lambda_transfer * transfer_loss
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update statistics
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                voice_input = batch['voice'].float().to(device)
                gesture_input = batch['gesture'].float().to(device)
                eye_input = batch['eye'].float().to(device)
                labels = batch['label'].long().to(device)
                
                # Forward pass
                outputs, voice_feat, gesture_feat, eye_feat, shared_rep = model(voice_input, gesture_input, eye_input)
                
                # Task loss
                task_loss = criterion(outputs, labels)
                
                # Transfer loss
                v_g_transfer = nn.functional.mse_loss(voice_feat, gesture_feat)
                v_e_transfer = nn.functional.mse_loss(voice_feat, eye_feat)
                g_e_transfer = nn.functional.mse_loss(gesture_feat, eye_feat)
                
                transfer_loss = v_g_transfer + v_e_transfer + g_e_transfer
                
                # Total loss
                loss = task_loss + lambda_transfer * transfer_loss
                
                # Update statistics
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print epoch statistics
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    return model

def main(use_synthetic_voice_gesture=True, eye_images_dir='hci_data\eye', num_samples=100, num_classes=5):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if use_synthetic_voice_gesture:
        print("Generating synthetic voice and gesture data...")
        data_dir = 'synthetic_data'
        label_file = generate_synthetic_data(num_samples=num_samples, num_classes=num_classes, data_dir=data_dir)
    else:
        # Use real data
        data_dir = 'E:\\BTech Project\\HCI Multimodal\\hci_data'
        label_file = 'E:\\BTech Project\\HCI Multimodal\\hci_data\\labels.csv'
    
    # Check if eye images directory exists
    if not os.path.exists(eye_images_dir):
        os.makedirs(eye_images_dir, exist_ok=True)
        print(f"Created directory for eye images: {eye_images_dir}")
        print("Please place your eye image files in this directory with naming format: P001_T001.jpg")
        print("After placing the images, run this script again.")
        return
    
    # Prepare dataset with eye images
    print("Preparing dataset with eye images...")
    voice_features, gesture_features, eye_features, labels = prepare_dataset(
        data_dir, label_file, eye_images_dir)
    
    # Print dataset information
    print(f"Dataset loaded: {len(labels)} samples")
    print(f"Voice features shape: {voice_features.shape}")
    print(f"Gesture features shape: {gesture_features.shape}")
    print(f"Eye features shape: {eye_features.shape}")
    print(f"Number of classes: {len(np.unique(labels))}")
    
    # Split data
    X_train_voice, X_test_voice, X_train_gesture, X_test_gesture, X_train_eye, X_test_eye, y_train, y_test = train_test_split(
        voice_features, gesture_features, eye_features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Create datasets
    train_dataset = HCIDataset(X_train_voice, X_train_gesture, X_train_eye, y_train)
    test_dataset = HCIDataset(X_test_voice, X_test_gesture, X_test_eye, y_test)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Get input dimensions
    voice_input_dim = X_train_voice.shape[1]
    gesture_input_dim = X_train_gesture.shape[1]
    eye_input_dim = X_train_eye.shape[1]
    
    # Get output dimension (number of classes)
    output_dim = len(np.unique(labels))
    
    # Create model
    model = CrossModalHCI(
        voice_input_dim=voice_input_dim,
        gesture_input_dim=gesture_input_dim,
        eye_input_dim=eye_input_dim,
        shared_dim=128,
        output_dim=output_dim
    )
    
    # Train model
    print("Training model...")
    model = train_model(model, train_loader, test_loader, 
                       num_epochs=10 if use_synthetic_voice_gesture else 30,
                       device=device)
    
    # Save model
    torch.save(model.state_dict(), 'cross_modal_hci_model.pth')
    print("Model saved successfully!")
    
    # Evaluate on test set
    model.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating on test set"):
            voice_input = batch['voice'].float().to(device)
            gesture_input = batch['gesture'].float().to(device)
            eye_input = batch['eye'].float().to(device)
            labels = batch['label'].long().to(device)
            
            # Forward pass
            outputs, _, _, _, _ = model(voice_input, gesture_input, eye_input)
            
            # Update statistics
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    
    test_acc = 100 * test_correct / test_total
    print(f"Test Accuracy: {test_acc:.2f}%")

if __name__ == "__main__":
    # Run with synthetic data by default
    main(use_synthetic_voice_gesture=False, num_samples=48, num_classes=6)