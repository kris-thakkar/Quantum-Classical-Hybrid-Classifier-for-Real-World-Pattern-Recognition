from flask import Flask, render_template, request
import numpy as np

# Qiskit / QML Imports
from qiskit_aer import AerSimulator
from qiskit.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.circuit.library import TwoLocal

from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit.algorithms.optimizers import COBYLA

# Scikit-Learn Imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

app = Flask(__name__)

@app.route("/")
def index():
    """
    Show a home page with a button to train the QNN on the XOR dataset.
    """
    return render_template("index.html")

@app.route("/train", methods=["POST"])
def train():
    """
    Train the QNN on the synthetic XOR data and display the results.
    """
    # -------------------------------------------------------
    # 1. Synthetic XOR Dataset (2 features, binary labels)
    # -------------------------------------------------------
    X_data = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ] * 10, dtype=float)  # repeated for a slightly bigger dataset

    # Map labels from {0, 1} to {-1, 1}
    y_data = np.array([-1, 1, 1, -1] * 10)  # XOR pattern repeated

    # Shuffle the dataset
    rng = np.random.default_rng(seed=42)
    indices = rng.permutation(len(X_data))
    X_data, y_data = X_data[indices], y_data[indices]

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=0.3, random_state=42
    )

    # -------------------------------------------------------
    # 2. Define Parametric Circuit with Data Embedding
    # -------------------------------------------------------
    num_qubits = 2
    feature_dim = 2  # matches X_data.shape[-1]

    input_params = ParameterVector('x', length=feature_dim)

    # Start with an empty circuit for 2 qubits
    qc = QuantumCircuit(num_qubits)

    # Encode each feature via a Ry rotation
    for i in range(num_qubits):
        qc.ry(np.pi * input_params[i], i)

    # Add a trainable ansatz on top
    ansatz = TwoLocal(
        num_qubits, 
        ['ry'], 
        'cz', 
        reps=1, 
        entanglement='full'
    )
    qc.compose(ansatz, inplace=True)

    # -------------------------------------------------------
    # 3. Create an Estimator Primitive (shots on AerSimulator)
    # -------------------------------------------------------
    simulator = AerSimulator()
    estimator = Estimator(
        backend=simulator,
        options={"shots": 1024}
    )

    # -------------------------------------------------------
    # 4. Define a 2-qubit SparsePauliOp for Observables
    # -------------------------------------------------------
    observables = [SparsePauliOp.from_list([('ZZ', 1.0)])]

    # -------------------------------------------------------
    # 5. Build an EstimatorQNN
    # -------------------------------------------------------
    qnn = EstimatorQNN(
        circuit=qc,
        observables=observables,
        input_params=input_params,
        weight_params=ansatz.parameters,
        estimator=estimator
    )

    # -------------------------------------------------------
    # 6. Wrap QNN in NeuralNetworkClassifier
    # -------------------------------------------------------
    classifier = NeuralNetworkClassifier(
        neural_network=qnn,
        optimizer=COBYLA(maxiter=50)
    )

    # -------------------------------------------------------
    # 7. Train and Evaluate with Label Mapping
    # -------------------------------------------------------
    classifier.fit(X_train, y_train)
    y_pred_raw = classifier.predict(X_test)

    # Map {-1, 1} to {0, 1}
    y_pred = ((y_pred_raw + 1) / 2).astype(int).flatten()

    # Ensure y_test is in {0, 1}
    y_test_mapped = ((y_test + 1) / 2).astype(int)

    accuracy = accuracy_score(y_test_mapped, y_pred)
    report = classification_report(y_test_mapped, y_pred)

    # Convert predictions & actual values to strings for display
    predictions_str = ", ".join(map(str, y_pred))
    actual_str = ", ".join(map(str, y_test_mapped))

    return render_template(
        "results.html",
        accuracy=accuracy,
        classification_report=report,
        predictions=predictions_str,
        actuals=actual_str
    )

if __name__ == "__main__":
    app.run(debug=True)
