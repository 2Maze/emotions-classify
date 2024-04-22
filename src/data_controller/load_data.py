

def load_data(bath_size=10, validation_bath_size=None, num_workers=1, test_size=0.15):
    validation_bath_size = validation_bath_size or bath_size

    dataset = EmotionDataset(annotation=join(ROOT_DIR, 'data', 'dataset', 'annotations.json'), padding_sec=PADDING_SEC)

    train_idx, val_idx = train_test_split(np.arange(len(dataset)),
                                          test_size=test_size,
                                          random_state=42,
                                          shuffle=True,
                                          stratify=dataset.emotion_labels)
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)

    train_dataloader = DataLoader(train_dataset, batch_size=bath_size, shuffle=True, collate_fn=collate_fn,
                                  num_workers=num_workers, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=validation_bath_size, shuffle=False, collate_fn=collate_fn,
                                num_workers=num_workers, drop_last=True)

    return train_dataloader, val_dataloader, dataset