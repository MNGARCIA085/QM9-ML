import torch
from torch_geometric.datasets import QM9


# sepaarte last 200 samples for test




def main():

	"""
	dataset = QM9(root="../data/qm9")

	N = 200
	indices = list(range(len(dataset) - N, len(dataset)))
	subset = [dataset[i] for i in indices]

	torch.save(subset, "../data/qm9_test_subset_last.pt")
	"""
	

	# test
	loaded = torch.load("../data/qm9_test_subset_last.pt", weights_only=False)
	print(len(loaded))   # 200



if __name__=="__main__":
	main()
