import torch
import torch.nn.functional as F
import time
import torch.nn as nn

from task import load_data, evaluate_model_accuracy, evaluate_model_f1_score, save_result
from my_extreme_learning_machine import MyExtremeLearningMachine
from my_ensemble_elm import MyEnsembleELM

def fit_elm_ls(model, train_loader):
    # Fit ELM with direct least-square solver
    #
    # Parameters:
    # model: The ELM model to be trained (class MyExtremeLearningMachine)
    # train_loader: DataLoader for training data
    #
    # Returns:
    # model: model with optimal hidden_to_output weight and bias
    # start_time: start training time
    # finish_time: finish training time
    #
    # Usage
    # Optimize ELM in task2a
    #
    # Reference: I use this code as a reference https://gist.github.com/derbydefi/616b1d5b986610f7cb5120ccaac85915

    start_time = time.time()  

    inputs =[]
    targets = []

    for input,target in train_loader:
        inputs.append(input)
        targets.append(target)

    inputs = torch.cat(inputs, dim=0)
    targets = torch.cat(targets, dim=0)

    with torch.no_grad():
        inputs = torch.flatten(inputs, 1)
        H = model.input_to_hidden(inputs) 
        H = torch.relu(H)

        T = F.one_hot(targets, num_classes=10).float()
        H_pseudo_inverse = torch.pinverse(H)
        beta = torch.matmul(H_pseudo_inverse, T)
        model.hidden_to_output.weight.copy_(beta.T)
    
    end_time = time.time()  

    return model, start_time, end_time

# ***** COPY FROM TASK.PY (MODIFIED) *****

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
    # start: start training time
    # finish: finish training time

    start_time = time.time()  

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
    
    end_time = time.time()  

    return model, start_time, end_time

def train(train_loader):
    # train model 5 and 6 with least square solver and SGD respectively
    #
    # Paremeter:
    # train_loader: DataLoader for training data
    #
    # Return:
    # [training time for least square solver, training time for SGD]

    print("MODEL 5: ELM WITH LEAST SQUARE SOLVER\n".center(40))
    print("=====================================\n")

    elm_least_square_model = MyExtremeLearningMachine()

    print(">> START TRAINING ELM USING LEAST SQUARE.....\n")
    elm_least_square_model, start_least_square, finish_least_square = fit_elm_ls(elm_least_square_model, train_loader)

    torch.save(elm_least_square_model.state_dict(), "elm_least_square_model_task2a.pth")
    print(">> DONE! MODEL SAVED TO: elm_least_square_model_task2a.pth\n")
    print("=====================================\n")

    print("MODEL 6: ELM ENSEMBLE\n")
    print("=====================================\n")

    ensemble1 = MyExtremeLearningMachine()
    ensemble2 = MyExtremeLearningMachine()
    ensemble3 = MyExtremeLearningMachine()

    print(">> ELM 1/3.....\n")
    ensemble1, start1, finish1 = fit_elm_sgd(ensemble1, train_loader)
    training_time1 = finish1 - start1
    torch.save(ensemble1.state_dict(), "ensemble1_task2a.pth")
    print(">> DONE! MODEL SAVED TO: ensemble1_task2a.pth\n")

    print(">> ELM 2/3.....\n")
    ensemble2, start2, finish2 = fit_elm_sgd(ensemble2, train_loader)
    training_time2 = finish2 - start2
    torch.save(ensemble2.state_dict(), "ensemble2_task2a.pth")
    print(">> DONE! MODEL SAVED TO: ensemble2_task2a.pth\n")

    print(">> ELM 3/3.....\n")
    ensemble3, start3, finish3 = fit_elm_sgd(ensemble3, train_loader)
    training_time3 = finish3 - start3
    torch.save(ensemble3.state_dict(), "ensemble3_task2a.pth")
    print(">> DONE! MODEL SAVED TO: ensemble3_task2a.pth\n")

    ensemble = MyEnsembleELM(models=[ensemble1, ensemble2, ensemble3])

    torch.save(ensemble.state_dict(), "elm_ensemble_model_task2a.pth")
    print(">> DONE! MODEL SAVED TO: elm_ensemble_model_task2a.pth\n")
    print("=====================================\n")

    return finish_least_square - start_least_square, training_time1 + training_time2 + training_time3

def main():
    train_loader, test_loader = load_data()

    print("=====================================\n")
    print("COURSEWORK 1 TASK 2A\n".center(40))
    print()
    print(">> THIS TASK HAVE TO TRAIN THE MODEL EVERYTIME EXECUTE SCRIPT\n")
    print(">> NOTE THAT YOU CAN ALWAYS SEE MY EXAMPLE OUTPUT AT task2a_example_output.txt\n".center(40))
    print("=====================================\n")

    training_time_least_square, training_time_ensemble = train(train_loader)

    elm_least_square = MyExtremeLearningMachine()
    elm_least_square.load_state_dict(torch.load("elm_least_square_model_task2a.pth", weights_only=True))

    print(">> EVALUATION: MODEL 5 ELM WITH LEAST SQUARE SOLVER\n")

    elm_least_square_correct, elm_least_square_total, elm_least_square_accuracy = evaluate_model_accuracy(elm_least_square, test_loader)
    elm_least_square_f1_score = evaluate_model_f1_score(elm_least_square, test_loader)
    print("=====================================\n")
    print(f">> CORRECT PREDICTION: {elm_least_square_correct}/{elm_least_square_total}\n")
    print(f">> ACCURACY: {elm_least_square_accuracy}%\n")
    print(f">> F1 SCORE: {elm_least_square_f1_score}\n")
    print(f">> TRAINING TIME: {training_time_least_square:.4f} SECONDS")
    print("=====================================\n")
    print(f">> SAVING RESULT TO new_result.png\n")
    save_result(elm_least_square, test_loader, filename="new_result.png")
    print(f">> DONE!\n")
    print("=====================================\n")

    elm1 = MyExtremeLearningMachine()
    elm2 = MyExtremeLearningMachine()
    elm3 = MyExtremeLearningMachine()

    elm1.load_state_dict(torch.load("ensemble1_task2a.pth", weights_only=True))
    elm2.load_state_dict(torch.load("ensemble2_task2a.pth", weights_only=True))
    elm3.load_state_dict(torch.load("ensemble3_task2a.pth", weights_only=True))

    ensemble = MyEnsembleELM(models=[elm1, elm2, elm3])
    ensemble.load_state_dict(torch.load("elm_ensemble_model_task2a.pth", weights_only=True))

    print(">> EVALUATION: MODEL 6 ENSEMBLE\n")
    ensemble_elm_correct, ensemble_elm_total, ensemble_elm_accuracy = evaluate_model_accuracy(ensemble, test_loader)
    ensemble_elm_f1_score = evaluate_model_f1_score(ensemble, test_loader)
    print("=====================================\n")
    print(f">> CORRECT PREDICTION: {ensemble_elm_correct}/{ensemble_elm_total}\n")
    print(f">> ACCURACY: {ensemble_elm_accuracy}%\n")
    print(f">> F1 SCORE: {ensemble_elm_f1_score}\n")
    print(f">> TRAINING TIME: {training_time_ensemble:.4f} SECONDS")
    print("=====================================\n")

    print("DISCUSSION\n".center(40))

    print(">> THE RESULTS OF MODEL 5 (ELM WITH LEAST SQUARES) ARE IMPRESSIVE!\n")
    print(">> IT TRAINS MUCH FASTER, TAKING ONLY ABOUT 20 SECONDS AND ACHIEVES HIGHER ACCURACY (AROUND 40–45%)\n")
    print(">> COMPARED TO ELM WITH SGD, WHICH TAKES 120 SECONDS AND YIELDS ONLY 35–40% ACCURACY.\n")
    print(">> THIS SUGGESTS THAT OPTIMIZING ELM USING A LEAST SQUARES SOLVER IS MORE EFFECTIVE THAN USING STANDARD SGD.\n")

    print("=====================================\n")

    print("RANDOM SEARCH\n".center(40))

    print("=====================================\n")
    print("MODEL 7: ELM WITH LEAST SQUARE SOLVER\n".center(40))
    print("=====================================\n")

    elm_least_square_model7 = MyExtremeLearningMachine(hidden_size=2000)
    print(">> START TRAINING ELM USING LEAST SQUARE.....\n")
    elm_least_square_model7, start_least_square7, finish_least_square7 = fit_elm_ls(elm_least_square_model7, train_loader)
    print(">> DONE.....\n")

    print("=====================================\n")
    print("MODEL 8: ELM WITH LEAST SQUARE SOLVER\n".center(40))
    print("=====================================\n")

    elm_least_square_model8 = MyExtremeLearningMachine(hidden_size=4000)
    print(">> START TRAINING ELM USING LEAST SQUARE.....\n")
    elm_least_square_model8, start_least_square8, finish_least_square8 = fit_elm_ls(elm_least_square_model8, train_loader)
    print(">> DONE.....\n")

    print("=====================================\n")

    print(">> COMPARE: MODEL 5,7,8 ELM WITH LEAST SQUARE SOLVER\n")

    _, _, elm_least_square_accuracy_model7 = evaluate_model_accuracy(elm_least_square_model7, test_loader)
    _, _, elm_least_square_accuracy_model8 = evaluate_model_accuracy(elm_least_square_model8, test_loader)

    print("=====================================\n")
    print(f">> ACCURACY MODEL 5 (hidden size 1000): {elm_least_square_accuracy}%\n")
    print(f">> ACCURACY MODEL 7 (hidden size 2000): {elm_least_square_accuracy_model7}%\n")
    print(f">> ACCURACY MODEL 8 (hidden size 4000): {elm_least_square_accuracy_model8}%\n")
    print(f">> CLAIM: HIGHER HIDDEN SIZE YIELD MORE ACCURACY\n")
    print("=====================================\n")

if __name__ == "__main__":
    main()