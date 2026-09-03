<?php
header('Content-Type: application/json');

$pythonExecutable = "python"; // <-- change this if needed
$scriptPath = __DIR__ . DIRECTORY_SEPARATOR . "python" . DIRECTORY_SEPARATOR . "predict_cli.py";

// Read the JSON body sent by script.js
$input = file_get_contents("php://input");
$data = json_decode($input, true);

if ($data === null) {
    http_response_code(400);
    echo json_encode(["error" => "Invalid JSON received by predict.php."]);
    exit;
}

$descriptorSpec = [
    0 => ["pipe", "r"], // stdin  — we write the JSON here
    1 => ["pipe", "w"], // stdout — we read the prediction here
    2 => ["pipe", "w"], // stderr — we read any Python errors here
];

$cmd = escapeshellarg($pythonExecutable) . ' ' . escapeshellarg($scriptPath);
$process = proc_open($cmd, $descriptorSpec, $pipes);

if (!is_resource($process)) {
    http_response_code(500);
    echo json_encode(["error" => "Failed to start the Python process. Check \$pythonExecutable in predict.php."]);
    exit;
}

fwrite($pipes[0], json_encode($data));
fclose($pipes[0]);

$output = stream_get_contents($pipes[1]);
fclose($pipes[1]);

$errorOutput = stream_get_contents($pipes[2]);
fclose($pipes[2]);

$exitCode = proc_close($process);

if (trim($output) === "") {
    http_response_code(500);
    echo json_encode([
        "error" => "The Python script produced no output.",
        "python_stderr" => $errorOutput,
        "hint" => "Run: pip install -r python/requirements.txt, and confirm \$pythonExecutable in predict.php works from a terminal.",
    ]);
    exit;
}

// Relay the Python script's JSON output as-is
echo $output;
