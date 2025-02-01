"""flower-project: A Flower / PyTorch app."""

from typing import List, Tuple

from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from imbalanced.net import Net, get_weights, set_weights, test
import torch
from sklearn.metrics import classification_report
from imbalanced.load_data import load_data
import random


global_model = Net()


def evaluate_fn(server_round, parameters_ndarrays, config):
    num_rounds = 3
    num_partitions = 4

    set_weights(global_model, parameters_ndarrays)

    partition_id = random.randint(0, num_partitions - 1) # randomly select a validation set
    _, valloader, testloader = load_data(partition_id, num_partitions, 32)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    global_model.to(device)

    if server_round == num_rounds:
        loss, accuracy = test(global_model, testloader, device)
        print(f"Server Round {server_round} Test Accuracy: {accuracy:.4f}")

    else:
        loss, accuracy = test(global_model, valloader, device)
        print(f"Server Round {server_round} Val Accuracy: {accuracy:.4f}")
    """
    criterion = torch.nn.CrossEntropyLoss().to(device) # add to device
    all_labels, all_preds = [], []
    correct, loss = 0, 0.0
    global_model.eval()
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = global_model(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
            preds = torch.max(outputs.data, 1)[1]
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    report = classification_report(all_labels, all_preds, target_names=["Normal", "Tuberculosis", "Pneumonia"])
    #print("server round " + str(server_round) + " server test" + report)
    accuracy = correct / len(testloader.dataset)
    print("server round " + str(server_round) + " test accurracy" + accuracy)
    loss = loss / len(testloader)
    """
    return loss, {"accuracy": accuracy, "loss": loss}


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    for i, (num_examples, m) in enumerate(metrics): # added log
        print(f"server_app.weighted_average: Client {i}: Examples={num_examples}, Accuracy={m['accuracy']:.4f}")

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

    # Initialize model parameters from the global model's weights
    parameters = ndarrays_to_parameters(get_weights(global_model))

    # Define strategy
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        evaluate_fn = evaluate_fn,
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=parameters,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
