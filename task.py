import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision.utils import make_grid, save_image
from PIL import Image, ImageDraw

from my_extreme_learning_machine import MyExtremeLearningMachine
from my_mix_up import MyMixUp
from my_ensemble_elm import MyEnsembleELM

def load_data():
    # Load CIFAR10
    #
    # Returns:
    # train_loader: DataLoader for training data
    # test_loader: DataLoader for testing data
    #
    # Usage
    # Load data for training

    tensor_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=tensor_transform)
    test_dataset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=tensor_transform)

    BATCH_SIZE = 40

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    return train_loader, test_loader

def fit_elm_sgd(model, train_loader, num_epochs=2, learning_rate=0.01):
    # Fit ELM with SGD
    #
    # Parameters:
    # model: The ELM model to be trained (class MyExtremeLearningMachine)
    # train_loader: DataLoader for training data
    # num_epochs: Number of epochs for training (default: 2)
    # learning_rate: Learning rate for SGD (default: 0.01)
    #
    # Returns:
    # model: model with optimal hidden_to_output weight and bias
    #
    # Usage
    # Train model 1 (ELM),3 (ENSEMBLE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.hidden_to_output.parameters(), lr=learning_rate)

    for epoch in range(1, num_epochs + 1):
        print(f"EPOCH {epoch}/{num_epochs}\n")
        print("----------")

        total_iterations = len(train_loader)
        count = 1

        print(f"ITERATION: 1/{total_iterations}")

        for inputs, labels in train_loader:
            if count % 100 == 0 or count == total_iterations:
                print(f"ITERATION: {count}/{total_iterations}")
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            count += 1
        print("----------\n")

    return model

def fit_elm_sgd_with_mixup(model, mixup, train_loader, num_epochs=2, learning_rate=0.01):
    # Fit ELM with SGD and MIXUP technique
    #
    # Parameters:
    # model: The ELM model to be trained (class MyExtremeLearningMachine)
    # mixup: object of class MyMixUp
    # train_loader: DataLoader for training data
    # num_epochs: Number of epochs for training (default: 2)
    # learning_rate: Learning rate for SGD (default: 0.01)
    #
    # Returns:
    # model: model with optimal hidden_to_output weight and bias
    #
    # Usage
    # Train model 2 (MIXUP),4 (ENSEMBLE + MIXUP)

    optimizer = torch.optim.SGD(model.hidden_to_output.parameters(), lr=learning_rate)

    for epoch in range(1, num_epochs + 1):
        print(f"EPOCH {epoch}/{num_epochs}\n")
        print("----------")

        total_iterations = len(train_loader)
        count = 1

        print(f"ITERATION: 1/{total_iterations}")

        for inputs, labels in train_loader:
            if count % 100 == 0 or count == total_iterations:
                print(f"ITERATION: {count}/{total_iterations}")
    
            mixed_inputs, mixed_labels = mixup.mix(inputs, labels)

            optimizer.zero_grad()
            outputs = model(mixed_inputs)
            loss = mixup.soft_cross_entropy(outputs, mixed_labels)
            loss.backward()
            optimizer.step()
            count += 1

        print("----------\n")

    return model

def evaluate_model_accuracy(model, test_loader):
    # Evaluate the accuracy of a given model on a test dataset.
    #
    # Parameter:
    # model: object of MyExtremeLearningMachine
    # test_loader: DataLoader object containing the test dataset.
    #
    # Return:
    # correct_predictions: The number of correctly predicted samples.
    # total_predictions: The total number of samples in the test dataset.
    # accuracy: The accuracy of the model as a percentage.
    #
    # Usage:
    # Evaluate model accuracy

    print(">> START EVALUATING THE MODEL ACCURACY.....\n")

    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            max_value, predicted_labels = torch.max(outputs, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted_labels == labels).sum().item()

    return correct_predictions, total_predictions, 100 * correct_predictions / total_predictions

def evaluate_model_f1_score(model, test_loader):
    # Evaluate the F1 score of a given model on a test dataset.
    #
    # Parameter:
    # model: object of MyExtremeLearningMachine
    # test_loader: DataLoader object containing the test dataset.
    #
    # Return:
    # f1_score: The average F1 score across all classes.
    #
    # Usage:
    # Evaluate model F1 score

    print(">> START EVALUATING THE MODEL F1 SCORE.....\n")

    num_classes = 10

    true_positives = [0] * num_classes
    false_positives = [0] * num_classes
    false_negatives = [0] * num_classes

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            max_value, predicted = torch.max(outputs, 1)

            for i in range(len(labels)):
                true_label = labels[i].item()
                pred_label = predicted[i].item()

                if pred_label == true_label:
                    true_positives[true_label] += 1
                else:
                    false_positives[pred_label] += 1
                    false_negatives[true_label] += 1

    f1_scores = []

    for i in range(num_classes):
        tp = true_positives[i]
        fp = false_positives[i]
        fn = false_negatives[i]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        f1_scores.append(f1)

    f1_score = sum(f1_scores) / num_classes

    return f1_score

def preview_mixup(train_loader, filename="mixup.png"):
    # Generates a preview of mixed images using the MixUp augmentation technique and saves the result as an image file.
    #
    # Parameters:
    # train_loader: DataLoader object providing batches of training data.
    # filename: Name of the file to save the preview image. Defaults to "mixup.png".
    #
    # Usage:
    # Save an example of how we mix images together

    print(f">> PREVIEWING MIXED IMAGES, SEE {filename}\n")
    
    mixup = MyMixUp()
    inputs, labels = next(iter(train_loader))
    mixed_inputs, mixed_labels = mixup.mix(inputs, labels)
    grid = make_grid(mixed_inputs[:16], nrow=4, normalize=True, padding=1)

    save_image(grid, filename)

    print(">> DONE!\n")
    print("=====================================\n")

def train(train_loader):
    # Train and save all 4 models
    #
    # Parameters:
    # train_loader: DataLoader object providing batches of training data.
    #
    # Usage:
    # train model and save model as .pth file
    # can omit if saved models exist

    print("MODEL 1: PURE ELM\n".center(40))
    print("=====================================\n")

    pure_elm_model = MyExtremeLearningMachine()

    print(">> START TRAINING ELM USING SGD.....\n")
    pure_elm_model = fit_elm_sgd(pure_elm_model, train_loader)

    torch.save(pure_elm_model.state_dict(), "elm_sgd_model.pth")
    print(">> DONE! MODEL SAVED TO: elm_sgd_model.pth\n")
    print("=====================================\n")
    print("MODEL 2: ELM + MIXUP\n".center(40))
    print("=====================================\n")

    mixup = MyMixUp()
    elm_with_mixup = MyExtremeLearningMachine()

    print(">> START TRAINING ELM WITH MIXUP.....\n")
    elm_with_mixup = fit_elm_sgd_with_mixup(elm_with_mixup, mixup, train_loader)

    torch.save(elm_with_mixup.state_dict(), "elm_sgd_mixup_model.pth")
    print(">> DONE! MODEL SAVED TO: elm_sgd_mixup_model.pth\n")
    print("=====================================\n")
    print("MODEL 3: ENSEMBLE ELM\n".center(40))
    print("=====================================\n")

    print(">> START TRAINING ELM WITH ENSEMBLE.....\n")

    # BEWARE: IF YOU ADD OR REMOVE THESE MODELS FROM THE ENSEMBLE
    # MAKE SURE YOU MODIFY CODE IN MAIN() AS WELL
    ensemble1 = MyExtremeLearningMachine()
    ensemble2 = MyExtremeLearningMachine()
    ensemble3 = MyExtremeLearningMachine()

    print(">> ELM 1/3.....\n")
    ensemble1 = fit_elm_sgd(ensemble1, train_loader)
    torch.save(ensemble1.state_dict(), "ensemble1.pth")
    print(">> DONE! MODEL SAVED TO: ensemble1.pth\n")

    print(">> ELM 2/3.....\n")
    ensemble2 = fit_elm_sgd(ensemble2, train_loader)
    torch.save(ensemble2.state_dict(), "ensemble2.pth")
    print(">> DONE! MODEL SAVED TO: ensemble2.pth\n")

    print(">> ELM 3/3.....\n")
    ensemble3 = fit_elm_sgd(ensemble3, train_loader)
    torch.save(ensemble3.state_dict(), "ensemble3.pth")
    print(">> DONE! MODEL SAVED TO: ensemble3.pth\n")

    ensemble_elm = MyEnsembleELM(models=[ensemble1, ensemble2, ensemble3])

    torch.save(ensemble_elm.state_dict(), "elm_ensemble_model.pth")
    print(">> DONE! MODEL SAVED TO: elm_ensemble_model.pth\n")
    print("=====================================\n")

    print("MODEL 4: ENSEMBLE ELM + MIXUP\n".center(40))
    print("=====================================\n")

    print(">> START TRAINING ENSEMBLE ELM WITH MIXUP.....\n")

    # BEWARE: IF YOU ADD OR REMOVE THESE MODELS FROM THE ENSEMBLE
    # MAKE SURE YOU MODIFY CODE IN MAIN() AS WELL
    ensemble4 = MyExtremeLearningMachine()
    ensemble5 = MyExtremeLearningMachine()
    ensemble6 = MyExtremeLearningMachine()

    print(">> ELM + MIXUP 1/3.....\n")
    ensemble4 = fit_elm_sgd_with_mixup(ensemble4, mixup, train_loader)
    torch.save(ensemble4.state_dict(), "ensemble4.pth")
    print(">> DONE! MODEL SAVED TO: ensemble4.pth\n")

    print(">> ELM + MIXUP 2/3.....\n")
    ensemble5 = fit_elm_sgd_with_mixup(ensemble5, mixup, train_loader)
    torch.save(ensemble5.state_dict(), "ensemble5.pth")
    print(">> DONE! MODEL SAVED TO: ensemble5.pth\n")

    print(">> ELM + MIXUP 3/3.....\n")
    ensemble6 = fit_elm_sgd_with_mixup(ensemble6, mixup, train_loader)
    torch.save(ensemble6.state_dict(), "ensemble6.pth")
    print(">> DONE! MODEL SAVED TO: ensemble6.pth\n")

    ensemble_elm_mixup = MyEnsembleELM(models=[ensemble4, ensemble5, ensemble6])

    torch.save(ensemble_elm_mixup.state_dict(), "elm_ensemble_mixup_model.pth")
    print(">> DONE! MODEL SAVED TO: elm_ensemble_mixup_model.pth\n")
    print("=====================================\n")

def save_result(optimal_model, test_loader, filename="result.png"):
    # Preview prediction result of the best model, save the result in result.png
    #
    # Parameters:
    # model: instance of MyExtremeLearningMachine or MyEnsembleELM
    # test_loader: DataLoader object providing batches of testing data.
    #
    # Usage:
    # preview prediction result

    class_names = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    images, labels = next(iter(test_loader))
    images, labels = images[:36], labels[:36]

    # Predict
    with torch.no_grad():
        outputs = optimal_model(images)
        max_values, predicts = torch.max(outputs, 1)

    # Unnormalize
    unnormalize = transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2])
    unnormalized_images = []
    for img in images:
        unnormalized_images.append(unnormalize(img))
    images = torch.stack(unnormalized_images)

    annotated_images = []

    for i in range(36):
        img = transforms.ToPILImage()(images[i].clamp(0, 1)).resize((96, 96))
        draw = ImageDraw.Draw(img)
        text = f"Actual: {class_names[labels[i]]}\nPredict: {class_names[predicts[i]]}"
        draw.text((4, 4), text, fill=(255, 255, 255))
        annotated_images.append(img)

    # Create blank canvas
    grid_img = Image.new("RGB", (96 * 6, 96 * 6))
    for idx, img in enumerate(annotated_images):
        row = idx // 6
        col = idx % 6
        grid_img.paste(img, (col * 96, row * 96))

    grid_img.save(filename)

def main():
    train_loader, test_loader = load_data()

    print("=====================================\n")
    print("COURSEWORK 1 TASK 2\n".center(40))
    print("=====================================\n")

    # **** If you wish to re-train all of my models, you can do it by uncomment this line
    # train(train_loader)

    preview_mixup(train_loader)

    # test models
    accuracies = []
    models = []

    # model 1: pure ELM
    pure_elm = MyExtremeLearningMachine()
    pure_elm.load_state_dict(torch.load("elm_sgd_model.pth", weights_only=True))
    print(">> EVALUATION: MODEL 1 PURE ELM\n")
    pure_elm_correct, pure_elm_total, pure_elm_accuracy = evaluate_model_accuracy(pure_elm, test_loader)
    pure_elm_f1_score = evaluate_model_f1_score(pure_elm, test_loader)
    print("=====================================\n")
    print(f">> CORRECT PREDICTION: {pure_elm_correct}/{pure_elm_total}\n")
    print(f">> ACCURACY: {pure_elm_accuracy}%\n")
    print(f">> F1 SCORE: {pure_elm_f1_score}\n")
    accuracies.append(pure_elm_accuracy)
    models.append(pure_elm)
    print("=====================================\n")

    # model 2: MixUp
    mix_up = MyExtremeLearningMachine()
    mix_up.load_state_dict(torch.load("elm_sgd_mixup_model.pth", weights_only=True))
    print(">> EVALUATION: MODEL 2 MIXUP\n")
    mixup_elm_correct, mixup_elm_total, mixup_elm_accuracy = evaluate_model_accuracy(mix_up, test_loader)
    mixup_elm_f1_score = evaluate_model_f1_score(mix_up, test_loader)
    print("=====================================\n")
    print(f">> CORRECT PREDICTION: {mixup_elm_correct}/{mixup_elm_total}\n")
    print(f">> ACCURACY: {mixup_elm_accuracy}%\n")
    print(f">> F1 SCORE: {mixup_elm_f1_score}\n")
    accuracies.append(mixup_elm_accuracy)
    models.append(mix_up)
    print("=====================================\n")

    # model 3: Ensemble
    elm1 = MyExtremeLearningMachine()
    elm2 = MyExtremeLearningMachine()
    elm3 = MyExtremeLearningMachine()
    elm1.load_state_dict(torch.load("ensemble1.pth", weights_only=True))
    elm2.load_state_dict(torch.load("ensemble2.pth", weights_only=True))
    elm3.load_state_dict(torch.load("ensemble3.pth", weights_only=True))
    ensemble = MyEnsembleELM(models=[elm1, elm2, elm3])
    ensemble.load_state_dict(torch.load("elm_ensemble_model.pth", weights_only=True))
    print(">> EVALUATION: MODEL 3 ENSEMBLE\n")
    ensemble_elm_correct, ensemble_elm_total, ensemble_elm_accuracy = evaluate_model_accuracy(ensemble, test_loader)
    ensemble_elm_f1_score = evaluate_model_f1_score(ensemble, test_loader)
    print("=====================================\n")
    print(f">> CORRECT PREDICTION: {ensemble_elm_correct}/{ensemble_elm_total}\n")
    print(f">> ACCURACY: {ensemble_elm_accuracy}%\n")
    print(f">> F1 SCORE: {ensemble_elm_f1_score}\n")
    accuracies.append(ensemble_elm_accuracy)
    models.append(ensemble)
    print("=====================================\n")

    # model 4: MIXUP + ENSEMBLE
    elm4 = MyExtremeLearningMachine()
    elm5 = MyExtremeLearningMachine()
    elm6 = MyExtremeLearningMachine()
    elm4.load_state_dict(torch.load("ensemble4.pth", weights_only=True))
    elm5.load_state_dict(torch.load("ensemble5.pth", weights_only=True))
    elm6.load_state_dict(torch.load("ensemble6.pth", weights_only=True))
    ensemble_mixup = MyEnsembleELM(models=[elm4, elm5, elm6])
    ensemble_mixup.load_state_dict(torch.load("elm_ensemble_mixup_model.pth", weights_only=True))
    print(">> EVALUATION: MODEL 4 MIXUP + ENSEMBLE\n")
    ensemble_elm_mixup_correct, ensemble_elm_mixup_total, ensemble_elm_mixup_accuracy = evaluate_model_accuracy(ensemble_mixup, test_loader)
    ensemble_elm_mixup_f1_score = evaluate_model_f1_score(ensemble_mixup, test_loader)
    print("=====================================\n")
    print(f">> CORRECT PREDICTION: {ensemble_elm_mixup_correct}/{ensemble_elm_mixup_total}\n")
    print(f">> ACCURACY: {ensemble_elm_mixup_accuracy}%\n")
    print(f">> F1 SCORE: {ensemble_elm_mixup_f1_score}\n")
    accuracies.append(ensemble_elm_mixup_accuracy)
    models.append(ensemble_mixup)
    print("=====================================\n")

    model_names = ["PURE ELM", "MIXUP", "ENSEMBLE", "MIXUP + ENSEMBLE"]
    optimal_model_index = accuracies.index(max(accuracies))
    optimal_model = models[optimal_model_index]
    print(f">> THE MODEL WITH HIGHEST AVERAGE ({max(accuracies)}) IS MODEL {optimal_model_index + 1} {model_names[optimal_model_index]}\n")
    print(f">> SAVING RESULT TO result.png\n")
    save_result(optimal_model, test_loader)
    print(f">> DONE!\n")
    print("=====================================\n")

    print("DISCUSSION\n".center(40))

    print("-------------------------------------\n")

    print("[Q1] WHAT IS CONSIDERED A 'RANDOM GUESS' IN A MULTICLASS CLASSIFICATION (100 WORDDS)\n")

    print(">> A RANDOM GUESS IN MULTICLASS CLASSIFICATION REFERS TO PREDICTING LABELS ENTIRELY AT RANDOM,\n")
    print(">> WITHOUT UTILIZING (LEARN) ANY INFORMATION FROM THE TRAINING DATA.\n")
    print(">> GAUHER(2016) CLAIMED THAT THE ACCURACY OF RANDOM GUESSING IN MULTICLASS CLASSIFICATION SHOULD EQUAL TO 1/K,\n")
    print(">> WHERE K IS THE NUMBER OF CLASSES. FROM THIS, I INFER THAT IF MY MODEL'S ACCURACY GREATER THAN 1/10 OR 10%,\n")
    print(">> IT IS LIKELY THAT IT PERFORMING BETTER THAN RANDOM GUESSING.\n")
    print(">> MOREOVER, SINCE WE ADD MORE DATA AUGMENTATION TECHNIQUES TO THE MODEL AND OBSERVING CHANGES IN ACCURACY (E.G., WITH ENSEMBLE METHODS)\n")
    print(">> THIS SUGGESTS THAT THE DATA PLAYS A ROLE IN THE MODEL'S PREDICTIONS.\n")
    print(">> REFERENCE: https://blog.revolutionanalytics.com/2016/03/classification-models.html\n")

    print()

    print("[Q2] JUSTIFY BRIEFLY YOUR METRICS IN A PRINTED COMMENT (50 WORDS)\n")

    print(">> BOTH F1-SCORE AND ACCURACY ARE REPORTED: ACCURACY IS SIMPLE AND INTUITIVE, WHILE F1-SCORE IS BETTER SUITED FOR IMBALANCED MULTICLASS DATA.\n")
    print(">> ALTHOUGH F1-SCORE MAY BE MORE APPROPRIATE FOR THIS CLASSIFICATION TASK, ACCURACY REMAINS USEFUL FOR CAPTURING OVERALL PERFORMANCE TRENDS.\n")
    print(">> NOTE THAT, ACCURACY IS USED TO COMPARE MODELS PERFORMANCE.\n")

    print("=====================================\n")

if __name__ == "__main__":
    main()