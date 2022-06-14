import argparse
import time

import torch
import tqdm

def create_cnn(kernels: int):
    net = torch.nn.Sequential(
        torch.nn.Conv2d(1, kernels, 7),
        torch.nn.AvgPool2d(2),
        torch.nn.Conv2d(kernels, kernels // 2, 7),
        torch.nn.AvgPool2d(2),
        torch.nn.Conv2d(kernels // 2, 3, 7),
        torch.nn.AvgPool2d(2),
        torch.nn.Flatten(1),
        torch.nn.Linear(3996, 1024),
        torch.nn.Dropout(0.2),
        torch.nn.Sigmoid(),
        torch.nn.Linear(1024, 10),
        torch.nn.Softmax(-1)
    )
    return net

def benchmark(net: torch.nn.Module, n: int, device: str):
    data = torch.empty(10, 1, 640, 480).to(device)

    runtimes = []
    for i in range(n):
        start = time.time_ns()
        net(data[n % 10])
        total = time.time_ns() - start
        runtimes.append(total)
    return torch.tensor(runtimes, dtype=torch.float64)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name", type=str)
    parser.add_argument("devices", nargs="*", default=["cpu"])
    args = parser.parse_args()

    output_file = f"output_{args.name}.dat"
    data = []
    for device in tqdm.tqdm(args.devices):
        for kernels in tqdm.tqdm([2 ** x for x in range(2, 6)]):
            net = create_cnn(kernels).to(device)
            for i in range(100):
                times = benchmark(net, i, device)
                datapoint = [device, kernels, i, times]
                data.append(datapoint)
                mean = times.mean()
                freq = 10e9 / mean

            # Occasionally save
            torch.save(data, output_file)