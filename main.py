from data.data import load_datasets

if __name__ == '__main__':
    train_dataset, test_dataset, train_dataloader, test_dataloader = load_datasets(64)

    for i, (img, label) in enumerate(train_dataloader):
        print(img.shape)
        break
	