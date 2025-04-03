# Dental Implant Classification - Experimental Testing Notebook
# This notebook enables systematic testing of data sources, processing techniques, and model architectures

# === 1. Setup and Imports ===

# Install any additional packages if needed
!pip install scikit-image tensorflow matplotlib seaborn pandas ipywidgets

# Import necessary libraries
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.applications import EfficientNetB3, ResNet50, DenseNet121
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import cv2
from skimage import io, color, exposure, filters, util
from google.colab import drive
import ipywidgets as widgets
from IPython.display import display, clear_output

# Mount Google Drive
drive.mount('/content/drive')

# Set base paths
BASE_PATH = '/content/drive/MyDrive/dental_implant_project'
DATA_PATH = os.path.join(BASE_PATH, 'data_collected')
RESULTS_PATH = os.path.join(BASE_PATH, 'results')

# Create results directory structure if it doesn't exist
os.makedirs(os.path.join(RESULTS_PATH, 'logs'), exist_ok=True)
os.makedirs(os.path.join(RESULTS_PATH, 'metrics'), exist_ok=True)
os.makedirs(os.path.join(RESULTS_PATH, 'models'), exist_ok=True)
os.makedirs(os.path.join(RESULTS_PATH, 'plots'), exist_ok=True)

# Configuration settings dictionary
config = {
    'data_source': None,
    'processing_method': 'original',
    'model_type': 'efficientnetb3',
    'learning_rate': 0.001,
    'batch_size': 16,
    'epochs': 10,
    'img_channels': 3,  # Will be set based on processing method
    'input_shape': None,  # Will be determined from data
    'num_classes': None,  # Will be determined from data
    'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
}

print("Setup complete!")

# === 2. Data Source Selection ===

def scan_data_sources():
    """Scan for available data sources in the data directory"""
    sources = []
    
    # Look for directories in the data_collected folder
    for item in os.listdir(DATA_PATH):
        source_path = os.path.join(DATA_PATH, item)
        if os.path.isdir(source_path):
            # Check if it contains train/val/test subdirectories
            if all(os.path.isdir(os.path.join(source_path, split)) for split in ['train', 'val', 'test']):
                sources.append(item)
    
    return sources

def get_class_distribution(data_dir):
    """Get number of images per class"""
    class_counts = {}
    
    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            count = len([f for f in os.listdir(class_path) if f.endswith(('.jpg', '.png', '.jpeg'))])
            class_counts[class_name] = count
    
    return class_counts

def display_source_info(source):
    """Display information about the selected data source"""
    source_path = os.path.join(DATA_PATH, source)
    
    # Get class distribution for each split
    splits = {}
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(source_path, split)
        class_counts = get_class_distribution(split_path)
        splits[split] = class_counts
    
    # Calculate total images
    total_images = sum(sum(counts.values()) for counts in splits.values())
    
    # Display summary
    print(f"=== Data Source: {source} ===")
    print(f"Total images: {total_images}")
    
    # Display class distribution
    for split, counts in splits.items():
        print(f"\n{split.capitalize()} set:")
        for class_name, count in counts.items():
            print(f"  {class_name}: {count} images")
    
    # Get image sizes
    train_path = os.path.join(source_path, 'train')
    class_dirs = [d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))]
    
    if class_dirs:
        first_class = class_dirs[0]
        class_path = os.path.join(train_path, first_class)
        image_files = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        if image_files:
            sample_img_path = os.path.join(class_path, image_files[0])
            img = plt.imread(sample_img_path)
            print(f"\nImage dimensions: {img.shape}")
            
            # Set the input shape based on the first image
            if len(img.shape) == 2:  # Grayscale
                config['input_shape'] = (img.shape[0], img.shape[1], 1)
                config['img_channels'] = 1
            else:  # RGB
                config['input_shape'] = img.shape
                config['img_channels'] = img.shape[2]
    
    # Display sample images
    display_sample_images(source)
    
    # Set number of classes
    config['num_classes'] = len(splits['train'])
    
    return splits

def display_sample_images(source, num_samples=3):
    """Display sample images from each class in the selected source"""
    train_path = os.path.join(DATA_PATH, source, 'train')
    classes = [d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))]
    
    num_classes = len(classes)
    fig, axes = plt.subplots(num_classes, num_samples, figsize=(15, 3*num_classes))
    
    # Handle the case where there's only one class
    if num_classes == 1:
        axes = [axes]
    
    for i, class_name in enumerate(classes):
        class_path = os.path.join(train_path, class_name)
        image_files = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        # Select random samples
        import random
        samples = random.sample(image_files, min(num_samples, len(image_files)))
        
        for j, file in enumerate(samples):
            img_path = os.path.join(class_path, file)
            img = plt.imread(img_path)
            
            # Handle grayscale vs RGB display
            if len(img.shape) == 2:
                axes[i][j].imshow(img, cmap='gray')
            else:
                axes[i][j].imshow(img)
                
            axes[i][j].set_title(f"{class_name}\n{file}")
            axes[i][j].axis('off')
    
    plt.tight_layout()
    plt.show()

def on_source_change(change):
    """Handle data source selection change"""
    if change['type'] == 'change' and change['name'] == 'value':
        selected_source = change['new']
        config['data_source'] = selected_source
        
        clear_output(wait=True)
        print(f"Selected data source: {selected_source}")
        
        # Display source information
        splits = display_source_info(selected_source)
        
        # Update processing method dropdown based on the new source
        update_processing_dropdown()

# Create dropdown for source selection
sources = scan_data_sources()
source_dropdown = widgets.Dropdown(
    options=sources,
    description='Data Source:',
    disabled=False,
)
source_dropdown.observe(on_source_change, names='value')

# Display the dropdown
display(source_dropdown)

def update_processing_dropdown():
    """Update the processing method dropdown based on current data"""
    # This will be called after source selection
    processing_dropdown.observe(on_processing_change, names='value')
    display(processing_dropdown)

# === 3. Image Processing Options ===

def apply_processing(img, method='original'):
    """Apply selected processing method to an image"""
    # Convert to float for processing if not already
    if img.dtype != np.float32 and img.dtype != np.float64:
        img = img.astype(np.float32) / 255.0
    
    if method == 'original':
        # No processing
        processed = img
    
    elif method == 'denoised':
        # Apply denoising
        if len(img.shape) == 3 and img.shape[2] == 3:  # RGB
            processed = cv2.fastNlMeansDenoisingColored(
                (img * 255).astype(np.uint8), None, 10, 10, 7, 21)
            processed = processed.astype(np.float32) / 255.0
        else:  # Grayscale
            if len(img.shape) == 3:  # Extra dimension
                img_gray = img[:,:,0]
            else:
                img_gray = img
            processed = cv2.fastNlMeansDenoising(
                (img_gray * 255).astype(np.uint8), None, 10, 7, 21)
            processed = processed.astype(np.float32) / 255.0
            
            # Restore shape if needed
            if len(img.shape) == 3 and processed.ndim == 2:
                processed = np.expand_dims(processed, axis=2)
    
    elif method == 'enhanced':
        # Apply contrast enhancement
        if len(img.shape) == 3 and img.shape[2] == 3:  # RGB
            img_lab = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(img_lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            processed_lab = cv2.merge((cl, a, b))
            processed = cv2.cvtColor(processed_lab, cv2.COLOR_LAB2RGB)
            processed = processed.astype(np.float32) / 255.0
        else:  # Grayscale
            if len(img.shape) == 3:  # Extra dimension
                img_gray = img[:,:,0]
            else:
                img_gray = img
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            processed = clahe.apply((img_gray * 255).astype(np.uint8))
            processed = processed.astype(np.float32) / 255.0
            
            # Restore shape if needed
            if len(img.shape) == 3 and processed.ndim == 2:
                processed = np.expand_dims(processed, axis=2)
    
    elif method == 'sharpened':
        # Apply sharpening
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])
        
        if len(img.shape) == 3 and img.shape[2] == 3:  # RGB
            processed = cv2.filter2D((img * 255).astype(np.uint8), -1, kernel)
            processed = processed.astype(np.float32) / 255.0
        else:  # Grayscale
            if len(img.shape) == 3:  # Extra dimension
                img_gray = img[:,:,0]
            else:
                img_gray = img
            processed = cv2.filter2D((img_gray * 255).astype(np.uint8), -1, kernel)
            processed = processed.astype(np.float32) / 255.0
            
            # Restore shape if needed
            if len(img.shape) == 3 and processed.ndim == 2:
                processed = np.expand_dims(processed, axis=2)
    
    # Ensure values are in [0, 1] range
    processed = np.clip(processed, 0, 1)
    
    return processed

def display_processing_comparison(source, method):
    """Display comparison of original and processed images"""
    if not source:
        print("Please select a data source first.")
        return
    
    train_path = os.path.join(DATA_PATH, source, 'train')
    classes = [d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))]
    
    # Select one image from each class
    fig, axes = plt.subplots(len(classes), 2, figsize=(10, 4*len(classes)))
    
    # Handle case with only one class
    if len(classes) == 1:
        axes = [axes]
    
    for i, class_name in enumerate(classes):
        class_path = os.path.join(train_path, class_name)
        image_files = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        if image_files:
            # Select first image
            img_path = os.path.join(class_path, image_files[0])
            img = plt.imread(img_path)
            
            # Convert to float for processing if not already
            if img.dtype != np.float32 and img.dtype != np.float64:
                img = img.astype(np.float32)
                if img.max() > 1.0:
                    img = img / 255.0
            
            # Apply processing
            processed_img = apply_processing(img, method)
            
            # Display original
            if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
                axes[i][0].imshow(img, cmap='gray')
            else:
                axes[i][0].imshow(img)
            axes[i][0].set_title(f"{class_name} - Original")
            axes[i][0].axis('off')
            
            # Display processed
            if len(processed_img.shape) == 2 or (len(processed_img.shape) == 3 and processed_img.shape[2] == 1):
                axes[i][1].imshow(processed_img, cmap='gray')
            else:
                axes[i][1].imshow(processed_img)
            axes[i][1].set_title(f"{class_name} - {method.capitalize()}")
            axes[i][1].axis('off')
    
    plt.tight_layout()
    plt.show()

def on_processing_change(change):
    """Handle processing method selection change"""
    if change['type'] == 'change' and change['name'] == 'value':
        selected_method = change['new']
        config['processing_method'] = selected_method
        
        clear_output(wait=True)
        print(f"Selected data source: {config['data_source']}")
        print(f"Selected processing method: {selected_method}")
        
        # Display comparison of original vs processed
        display_processing_comparison(config['data_source'], selected_method)
        
        # Update model selection options
        update_model_dropdown()

# Create dropdown for processing selection
processing_dropdown = widgets.Dropdown(
    options=['original', 'denoised', 'enhanced', 'sharpened'],
    description='Processing:',
    disabled=False,
)

# === 4. Model Selection ===

def build_efficientnetb3_model(input_shape, num_classes):
    """Build EfficientNetB3 model"""
    # Use transfer learning for efficient training
    base_model = EfficientNetB3(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Freeze base model for initial training
    base_model.trainable = False
    
    # Create classification head
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    return model

def build_custom_cnn_model(input_shape, num_classes):
    """Build custom CNN model"""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    return model

def build_resnet50_model(input_shape, num_classes):
    """Build ResNet50 model"""
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Freeze base model for initial training
    base_model.trainable = False
    
    # Create classification head
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    return model

def build_densenet121_model(input_shape, num_classes):
    """Build DenseNet121 model"""
    base_model = DenseNet121(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Freeze base model for initial training
    base_model.trainable = False
    
    # Create classification head
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    return model

def get_model(model_type, input_shape, num_classes):
    """Get the specified model architecture"""
    if model_type == 'efficientnetb3':
        return build_efficientnetb3_model(input_shape, num_classes)
    elif model_type == 'custom_cnn':
        return build_custom_cnn_model(input_shape, num_classes)
    elif model_type == 'resnet50':
        return build_resnet50_model(input_shape, num_classes)
    elif model_type == 'densenet121':
        return build_densenet121_model(input_shape, num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def on_model_change(change):
    """Handle model selection change"""
    if change['type'] == 'change' and change['name'] == 'value':
        selected_model = change['new']
        config['model_type'] = selected_model
        
        clear_output(wait=True)
        print(f"Selected data source: {config['data_source']}")
        print(f"Selected processing method: {config['processing_method']}")
        print(f"Selected model: {selected_model}")
        
        # Display hyperparameter controls
        display_hyperparameter_controls()

def update_model_dropdown():
    """Update model dropdown based on processing method"""
    model_dropdown.observe(on_model_change, names='value')
    display(model_dropdown)

# Create dropdown for model selection
model_dropdown = widgets.Dropdown(
    options=['efficientnetb3', 'custom_cnn', 'resnet50', 'densenet121'],
    description='Model:',
    disabled=False,
)

# === 5. Hyperparameter Controls ===

def display_hyperparameter_controls():
    """Display sliders for hyperparameter configuration"""
    # Learning rate slider
    lr_slider = widgets.FloatLogSlider(
        value=1e-3,
        base=10,
        min=-5,  # 1e-5
        max=-1,  # 1e-1
        step=0.2,
        description='Learning Rate:',
        continuous_update=False
    )
    
    # Batch size slider
    batch_slider = widgets.IntSlider(
        value=16,
        min=4,
        max=64,
        step=4,
        description='Batch Size:',
        continuous_update=False
    )
    
    # Epochs slider
    epochs_slider = widgets.IntSlider(
        value=10,
        min=1,
        max=50,
        step=1,
        description='Epochs:',
        continuous_update=False
    )
    
    # Training button
    train_button = widgets.Button(
        description='Start Training',
        button_style='success',
        icon='play'
    )
    
    # Update config when sliders change
    def on_lr_change(change):
        config['learning_rate'] = change['new']
        print(f"Learning rate set to: {change['new']}")
    
    def on_batch_change(change):
        config['batch_size'] = change['new']
        print(f"Batch size set to: {change['new']}")
    
    def on_epochs_change(change):
        config['epochs'] = change['new']
        print(f"Epochs set to: {change['new']}")
    
    def on_train_click(b):
        clear_output(wait=True)
        print("Starting training with the following configuration:")
        print(f"- Data source: {config['data_source']}")
        print(f"- Processing method: {config['processing_method']}")
        print(f"- Model: {config['model_type']}")
        print(f"- Learning rate: {config['learning_rate']}")
        print(f"- Batch size: {config['batch_size']}")
        print(f"- Epochs: {config['epochs']}")
        print(f"- Input shape: {config['input_shape']}")
        print(f"- Number of classes: {config['num_classes']}")
        
        # Start training
        train_model()
    
    lr_slider.observe(on_lr_change, names='value')
    batch_slider.observe(on_batch_change, names='value')
    epochs_slider.observe(on_epochs_change, names='value')
    train_button.on_click(on_train_click)
    
    # Display controls
    display(lr_slider, batch_slider, epochs_slider, train_button)

# === 6. Data Loading & Processing ===

def create_data_generators():
    """Create data generators with processing"""
    if not config['data_source']:
        print("Please select a data source.")
        return None, None, None
    
    # Paths to data splits
    train_dir = os.path.join(DATA_PATH, config['data_source'], 'train')
    val_dir = os.path.join(DATA_PATH, config['data_source'], 'val')
    test_dir = os.path.join(DATA_PATH, config['data_source'], 'test')
    
    # Define preprocessing function based on selected method
    def preprocess_fn(img):
        return apply_processing(img, config['processing_method'])
    
    # Create data generators
    if config['processing_method'] == 'original':
        # For original images, use standard rescaling
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            validation_split=0.0  # We have a separate validation set
        )
        
        val_datagen = ImageDataGenerator(rescale=1./255)
        test_datagen = ImageDataGenerator(rescale=1./255)
    else:
        # For other processing methods, use preprocessing function
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_fn,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            validation_split=0.0
        )
        
        val_datagen = ImageDataGenerator(preprocessing_function=preprocess_fn)
        test_datagen = ImageDataGenerator(preprocessing_function=preprocess_fn)
    
    # Create generators
    color_mode = 'grayscale' if config['img_channels'] == 1 else 'rgb'
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(config['input_shape'][0], config['input_shape'][1]),
        batch_size=config['batch_size'],
        class_mode='categorical',
        color_mode=color_mode,
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(config['input_shape'][0], config['input_shape'][1]),
        batch_size=config['batch_size'],
        class_mode='categorical',
        color_mode=color_mode,
        shuffle=False
    )
    
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(config['input_shape'][0], config['input_shape'][1]),
        batch_size=config['batch_size'],
        class_mode='categorical',
        color_mode=color_mode,
        shuffle=False
    )
    
    # Update class indices in config
    config['class_indices'] = train_generator.class_indices
    
    return train_generator, val_generator, test_generator

# === 7. Model Training ===

def train_model():
    """Train the model with selected configuration"""
    # Create data generators
    train_generator, val_generator, test_generator = create_data_generators()
    
    if not train_generator:
        return
    
    # Build model
    model = get_model(
        model_type=config['model_type'],
        input_shape=config['input_shape'],
        num_classes=config['num_classes']
    )
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=config['learning_rate']),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Create results subdirectory
    timestamp = config['timestamp']
    experiment_name = f"{config['data_source']}_{config['processing_method']}_{config['model_type']}_{timestamp}"
    experiment_dir = os.path.join(RESULTS_PATH, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Create callbacks
    model_checkpoint = ModelCheckpoint(
        os.path.join(RESULTS_PATH, 'models', f"{experiment_name}_best.keras"),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    # Train model
    print("\nStarting training...")
    history = model.fit(
        train_generator,
        epochs=config['epochs'],
        validation_data=val_generator,
        callbacks=[model_checkpoint, early_stopping, reduce_lr],
        verbose=1
    )
    
    # Save final model
    final_model_path = os.path.join(RESULTS_PATH, 'models', f"{experiment_name}_final.keras")
    model.save(final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")
    
    # Plot training history
    plot_training_history(history, experiment_name)
    
    # Evaluate model
    evaluate_model(model, test_generator, experiment_name)
    
    # Save configuration
    save_config(experiment_name)
    
    return model, history

def plot_training_history(history, experiment_name):
    """Plot and save training history"""
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    
    # Save plot
    plt_path = os.path.join(RESULTS_PATH, 'plots', f"{experiment_name}_history.png")
    plt.savefig(plt_path)
    plt.show()
    
    # Save history data
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(os.path.join(RESULTS_PATH, 'logs', f"{experiment_name}_history.csv"))

# === 8. Model Evaluation ===

def evaluate_model(model, test_generator, experiment_name):
    """Evaluate model on test set and save metrics"""
    print("\nEvaluating model on test set...")
    test_loss, test_acc = model.evaluate(test_generator, verbose=1)
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    # Generate predictions
    predictions = model.predict(test_generator)
    y_pred = np.argmax(predictions, axis=1)
    
    # True labels
    y_true = test_generator.classes
    
    # Get class names
    class_names = list(test_generator.class_indices.keys())
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # Save confusion matrix
    cm_path = os.path.join(RESULTS_PATH, 'plots', f"{experiment_name}_confusion_matrix.png")
    plt.savefig(cm_path)
    plt.show()
    
    # Generate classification report
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    
    # Save report
    report_path = os.path.join(RESULTS_PATH, 'metrics', f"{experiment_name}_classification_report.csv")
    report_df.to_csv(report_path)
    
    # Display report
    print("\nClassification Report:")
    print(pd.DataFrame(report).transpose())
    
    # Save test metrics
    metrics = {
        'test_accuracy': test_acc,
        'test_loss': test_loss,
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }
    
    import json
    with open(os.path.join(RESULTS_PATH, 'metrics', f"{experiment_name}_metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Plot ROC curves for multi-class
    plot_roc_curves(y_true, predictions, class_names, experiment_name)

def plot_roc_curves(y_true, y_pred_proba, class_names, experiment_name):
    """Plot ROC curves for each class"""
    plt.figure(figsize=(10, 8))
    
    # One-hot encode true labels
    from sklearn.preprocessing import label_binarize
    y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(len(class_names)):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
        plt.plot(fpr[i], tpr[i], lw=2,
                 label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')
    
    # Plot random guess line
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    
    # Save ROC curves
    roc_path = os.path.join(RESULTS_PATH, 'plots', f"{experiment_name}_roc_curves.png")
    plt.savefig(roc_path)
    plt.show()

def save_config(experiment_name):
    """Save experiment configuration"""
    config_copy = config.copy()
    
    # Convert input_shape to list for JSON serialization
    if config_copy['input_shape'] is not None:
        config_copy['input_shape'] = list(config_copy['input_shape'])
    
    # Save config
    import json
    with open(os.path.join(RESULTS_PATH, 'logs', f"{experiment_name}_config.json"), 'w') as f:
        json.dump(config_copy, f, indent=4)

# === 9. Feature Visualization ===

def visualize_model_attention(model, img_path, experiment_name):
    """Visualize what the model is focusing on using Grad-CAM"""
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize image to match input shape
    img_resized = cv2.resize(img, (config['input_shape'][1], config['input_shape'][0]))
    
    # Preprocess image
    if config['img_channels'] == 1:
        # Convert to grayscale if needed
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
        img_resized = np.expand_dims(img_resized, axis=-1)
    
    # Apply selected processing
    img_processed = apply_processing(img_resized / 255.0, config['processing_method'])
    
    # Add batch dimension
    img_batch = np.expand_dims(img_processed, axis=0)
    
    # Get model prediction
    predictions = model.predict(img_batch)
    predicted_class = np.argmax(predictions[0])
    
    # Find the last convolutional layer
    last_conv_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer
            break
    
    if last_conv_layer is None:
        print("Could not find convolutional layer for visualization.")
        return
    
    # Create Grad-CAM
    grad_model = Model(
        inputs=[model.inputs],
        outputs=[last_conv_layer.output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_batch)
        class_output = predictions[:, predicted_class]
    
    grads = tape.gradient(class_output, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    conv_outputs = conv_outputs[0]
    
    for i in range(pooled_grads.shape[0]):
        conv_outputs[:, :, i] *= pooled_grads[i]
    
    heatmap = tf.reduce_mean(conv_outputs, axis=-1)
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    heatmap = heatmap.numpy()
    
    # Resize heatmap to original image size
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # Apply colormap to heatmap
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Overlay heatmap on original image
    alpha = 0.4
    superimposed_img = cv2.addWeighted(
        cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
        1 - alpha,
        heatmap,
        alpha,
        0
    )
    
    # Display original and heatmap overlay
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
    plt.title('Grad-CAM Heatmap')
    plt.axis('off')
    
    plt.tight_layout()
    
    # Save visualization
    viz_path = os.path.join(RESULTS_PATH, 'plots', f"{experiment_name}_gradcam.png")
    plt.savefig(viz_path)
    plt.show()
    
    return heatmap, superimposed_img

# === 10. Experiment Comparison ===

def compare_experiments(experiment_names):
    """Compare results from multiple experiments"""
    if not experiment_names or len(experiment_names) < 2:
        print("Please provide at least two experiment names to compare.")
        return
    
    metrics_data = []
    
    for name in experiment_names:
        metrics_path = os.path.join(RESULTS_PATH, 'metrics', f"{name}_metrics.json")
        if not os.path.exists(metrics_path):
            print(f"Metrics file not found for experiment: {name}")
            continue
        
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        # Load config
        config_path = os.path.join(RESULTS_PATH, 'logs', f"{name}_config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                exp_config = json.load(f)
        else:
            exp_config = {}
        
        # Extract key information
        experiment_info = {
            'name': name,
            'source': exp_config.get('data_source', 'unknown'),
            'processing': exp_config.get('processing_method', 'unknown'),
            'model': exp_config.get('model_type', 'unknown'),
            'accuracy': metrics.get('test_accuracy', 0),
            'loss': metrics.get('test_loss', 0)
        }
        
        # Extract f1-scores from classification report
        if 'classification_report' in metrics:
            report = metrics['classification_report']
            if 'weighted avg' in report:
                experiment_info['f1_score'] = report['weighted avg']['f1-score']
                experiment_info['precision'] = report['weighted avg']['precision']
                experiment_info['recall'] = report['weighted avg']['recall']
        
        metrics_data.append(experiment_info)
    
    # Create dataframe for comparison
    df = pd.DataFrame(metrics_data)
    
    # Display comparison table
    print("=== Experiment Comparison ===")
    print(df[['name', 'source', 'processing', 'model', 'accuracy', 'f1_score']])
    
    # Plot comparison
    plt.figure(figsize=(14, 8))
    
    plt.subplot(2, 2, 1)
    sns.barplot(x='name', y='accuracy', data=df)
    plt.title('Test Accuracy Comparison')
    plt.xticks(rotation=45)
    
    plt.subplot(2, 2, 2)
    sns.barplot(x='name', y='f1_score', data=df)
    plt.title('F1 Score Comparison')
    plt.xticks(rotation=45)
    
    plt.subplot(2, 2, 3)
    sns.barplot(x='name', y='precision', data=df)
    plt.title('Precision Comparison')
    plt.xticks(rotation=45)
    
    plt.subplot(2, 2, 4)
    sns.barplot(x='name', y='recall', data=df)
    plt.title('Recall Comparison')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # Save comparison plot
    comparison_path = os.path.join(RESULTS_PATH, 'plots', f"experiment_comparison.png")
    plt.savefig(comparison_path)
    plt.show()
    
    return df

# Add a helper function to list available experiments
def list_experiments():
    """List all available experiments for comparison"""
    metric_files = [f for f in os.listdir(os.path.join(RESULTS_PATH, 'metrics')) 
                   if f.endswith('_metrics.json')]
    
    experiment_names = [f.replace('_metrics.json', '') for f in metric_files]
    return sorted(experiment_names)

# Display compare experiments button
def display_comparison_controls():
    """Display controls for experiment comparison"""
    experiments = list_experiments()
    
    if not experiments:
        print("No experiments found to compare.")
        return
    
    # Create multi-select widget
    experiment_selector = widgets.SelectMultiple(
        options=experiments,
        description='Experiments:',
        disabled=False
    )
    
    # Compare button
    compare_button = widgets.Button(
        description='Compare Selected',
        button_style='info',
        icon='chart-bar'
    )
    
    def on_compare_click(b):
        selected = experiment_selector.value
        if not selected or len(selected) < 2:
            print("Please select at least two experiments to compare.")
            return
        
        compare_experiments(selected)
    
    compare_button.on_click(on_compare_click)
    
    # Display widgets
    display(experiment_selector, compare_button)

print("Notebook ready! Start by selecting a data source.")