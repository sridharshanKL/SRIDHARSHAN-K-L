import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from unet_model import unet_model

IMG_SIZE = 256
image_dir = "data/images/"
mask_dir = "data/masks/"

def load_data(image_dir, mask_dir, img_size=IMG_SIZE):
    images, masks = [], []
    for filename in os.listdir(image_dir):
        img_path = os.path.join(image_dir, filename)
        mask_path = os.path.join(mask_dir, filename)
        if os.path.exists(img_path) and os.path.exists(mask_path):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if img is not None and mask is not None:
                img = cv2.resize(img, (img_size, img_size)) / 255.0
                mask = cv2.resize(mask, (img_size, img_size)) / 255.0
                mask = np.expand_dims(mask, axis=-1)
                images.append(img)
                masks.append(mask)
            else:
                print(f"Could not load image or mask: {filename}")
        else:
            print(f"Missing file: {filename}")
    print(f"Total loaded: {len(images)} pairs")
    return np.array(images).reshape(-1, img_size, img_size, 1), np.array(masks)


X, y = load_data(image_dir, mask_dir)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)

model = unet_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

callbacks = [
    ModelCheckpoint("tumor_model.h5", save_best_only=True),
    EarlyStopping(patience=5, restore_best_weights=True)
]

'''history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=8,
    callbacks=callbacks
)'''
model.load_weights("tumor_model.h5")
preds = model.predict(X_val)

i = 0
plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.imshow(X_val[i].squeeze(), cmap='gray')
plt.title('Original MRI')
plt.subplot(1,3,2)
plt.imshow(y_val[i].squeeze(), cmap='gray')
plt.title('Ground Truth')
plt.subplot(1,3,3)
plt.imshow(preds[i].squeeze(), cmap='gray')
plt.title('Predicted Mask')
plt.show()
