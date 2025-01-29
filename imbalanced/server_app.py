"""flower-project: A Flower / PyTorch app."""

from typing import List, Tuple

from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from imbalanced.net import Net, get_weights, set_weights
import torch
from sklearn.metrics import classification_report
from imbalanced.load_data import load_data


def evaluate_fn(server_round, parameters_ndarrays, config):
    net = Net()
    set_weights(net, parameters_ndarrays)
    def get_evaluate_fn(net):
        _, _, testloader = load_data(1, 3, 32)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        """Validate the model on the test set."""
        net.to(device)
        criterion = torch.nn.CrossEntropyLoss().to(device) # add to device
        all_labels, all_preds = [], []
        correct, loss = 0, 0.0
        net.eval()
        with torch.no_grad():
            for batch in testloader:
                images = batch["img"].to(device)
                labels = batch["label"].to(device)
                outputs = net(images)
                loss += criterion(outputs, labels).item()
                correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
                preds = torch.max(outputs.data, 1)[1]
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
        report = classification_report(all_labels, all_preds, target_names=["Normal", "Tuberculosis", "Pneumonia"])
        print(str(server_round) + "server test" + report)
        accuracy = correct / len(testloader.dataset)
        loss = loss / len(testloader)
        return loss, accuracy
    
    loss, accuracy = get_evaluate_fn(net)
    #return {"loss": loss, "accuracy": accuracy}
    return loss, {"accuracy": accuracy, "loss": loss}

"""
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    for i, (num_examples, m) in enumerate(metrics): # added log
        print(f"server_app.weighted_average: Client {i}: Examples={num_examples}, Accuracy={m['accuracy']:.4f}")

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}
"""

def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

    # Initialize model parameters
    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)

    # Define strategy
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        evaluate_fn = evaluate_fn,
        #evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=parameters,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
