import numpy as np
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, precision_score
import wandb
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

def build_model(num_layers, num_classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))

    for _ in range(num_layers-1):
        model.add(Conv2D(32, (3, 3), activation='relu'))

    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    return model

def kfold_cross_validation(num_layers, learning_rate, k):
    # Step 1: Load the CIFAR-10 dataset
    # Split the dataset
    num_classes = 10
    epochs = 10
    batch_size = 32
    class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # Step 2: Preprocess the data
    # Normalize the data
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # One-hot encode the labels
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    # Step 3: Perform k-fold cross-validation
    kf = KFold(n_splits=k, shuffle=True)
    # Initialize a new run with a unique name
    run_name = f"{k}folds_{num_layers}_layers_{learning_rate}_learning_rate"
    wandb.init(
        project="cifar10-kfold-cnn-group-project",
        name=run_name,
        config={
            "learning_rate": learning_rate,
            "architecture": "CNN",
            "dataset": "CIFAR-10",
            "epochs": epochs,
            "kfolds": k,
            "num_layers": num_layers
        }
    )
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    
        # Build the CNN model
        model = build_model(num_layers, num_classes)

        # Compile the model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Train the model
        early_stopping = EarlyStopping(monitor='val_loss', patience=3)
        history = model.fit(
            X_train[train_idx],
            y_train[train_idx],
            validation_data=(X_train[val_idx], y_train[val_idx]),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[early_stopping]  # Add the early stopping callback
        )
         # Log every model after training on a fold
        wandb.log({"model_summary": model.summary()})

    # Save the final model after training on all the folds
    model.save(f"{run_name}.h5")


    # Log accuracy vs epoch
    accuracy_vs_epochs = wandb.plot.line_series(
        xs=np.linspace(0, len(history.history['accuracy']), num=len(history.history['accuracy'])),
        ys=[history.history['accuracy'], history.history['val_accuracy']],
        keys=['train', 'val'],
        title="Accuracy vs. epochs",
        xname="Epochs"
    )
    wandb.log({f"accuracy_vs_epochs_{run_name}": accuracy_vs_epochs})

    # Log loss vs epoch
    loss_vs_epochs = wandb.plot.line_series(
        xs=np.linspace(0, len(history.history['loss']), num=len(history.history['loss'])),
        ys=[history.history['loss'], history.history['val_loss']],
        keys=['train', 'val'],
        title="Loss vs. epochs",
        xname="Epochs"
    )
    wandb.log({f"loss_vs_epochs_{run_name}": loss_vs_epochs})
        
    # Plot and save accuracy vs epoch
    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Accuracy vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='lower right')
    plt.savefig(f"accuracy_vs_epochs_{run_name}.png")
    plt.close()

    # Plot and save loss vs epoch
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.savefig(f"loss_vs_epochs_{run_name}.png")
    plt.close()
    
    # Step 4: Evaluate the model on the test set
    test_f1_score = f1_score(y_test.argmax(axis=1), model.predict(X_test).argmax(axis=1), average='micro')
    test_accuracy = accuracy_score(y_test.argmax(axis=1), model.predict(X_test).argmax(axis=1))
    test_precision = precision_score(y_test.argmax(axis=1), model.predict(X_test).argmax(axis=1), average='micro')
    wandb.log({"test_f1_score": test_f1_score, "test_accuracy": test_accuracy, "test_precision": test_precision})
    
    cm = wandb.plot.confusion_matrix(
        y_true=y_test.argmax(axis=1),
        preds=model.predict(X_test).argmax(axis=1),
        class_names=class_names
    )
    wandb.log({f"confusion_matrix_{run_name}": cm})
    
    # precision_micro vs recall_micro
    custom_chart = wandb.plot.pr_curve(
        y_true=y_test.argmax(axis=1),
        y_probas=model.predict(X_test),
        labels=class_names
    )
    wandb.log({f"pr_curve_{run_name}": custom_chart})
    
    # Log some images and their predictions
    for i in range(10):
        wandb.log({
            "image": wandb.Image(X_test[i], caption=f"Predicted: {class_names[model.predict(X_test[i].reshape(1,32,32,3)).argmax()]}, Actual: {class_names[y_test[i].argmax()]}")
        })

if __name__ == "__main__":
    num_layers=3
    learning_rate=0.001
    k=5
    kfold_cross_validation(num_layers, learning_rate, k)
