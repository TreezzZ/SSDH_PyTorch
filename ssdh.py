import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from scipy.spatial.distance import pdist, squareform
from scipy.stats import norm
from loguru import logger

from model_loader import load_model
from evaluate import mean_average_precision


def train(train_dataloader,
          query_dataloader,
          retrieval_dataloader,
          multi_labels,
          code_length,
          num_features,
          alpha,
          beta,
          max_iter,
          arch,
          lr,
          device,
          verbose,
          evaluate_interval,
          snapshot_interval,
          topk,
          checkpoint=None,
          ):
    """
    Training model.

    Args
        train_dataloader(torch.evaluate.data.DataLoader): Training data loader.
        query_dataloader(torch.evaluate.data.DataLoader): Query data loader.
        retrieval_dataloader(torch.evaluate.data.DataLoader): Retrieval data loader.
        multi_labels(bool): True, if dataset is multi-labels.
        code_length(int): Hash code length.
        num_features(int): Number of features.
        alpha, beta(float): Hyper-parameters.
        max_iter(int): Number of iterations.
        arch(str): Model name.
        lr(float): Learning rate.
        device(torch.device): GPU or CPU.
        verbose(bool): Print log.
        evaluate_interval(int): Interval of evaluation.
        snapshot_interval(int): Interval of snapshot.
        topk(int): Calculate top k data points map.
        checkpoint(str, optional): Paht of checkpoint.

    Returns
        None
    """
    # Model, optimizer, criterion
    model = load_model(arch, code_length)
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5)
    criterion = SSDH_Loss()

    # Resume
    resume_it = 0
    if checkpoint:
        optimizer, resume_it = model.load_snapshot(checkpoint, optimizer)
        logger.info('[resume:{}][iteration:{}]'.format(checkpoint, resume_it+1))

    # Extract features
    features = extract_features(model, train_dataloader, num_features, device, verbose)

    # Generate similarity matrix
    S = generate_similarity_matrix(features, alpha, beta).to(device)

    # Training
    model.train()
    for epoch in range(resume_it, max_iter):
        n_batch = len(train_dataloader)
        for i, (data, _, index) in enumerate(train_dataloader):
            # Current iteration
            cur_iter = epoch * n_batch + i + 1

            data = data.to(device)
            optimizer.zero_grad()

            v = model(data)
            H = v @ v.t() / code_length
            targets = S[index, :][:, index]
            loss = criterion(H, targets)

            loss.backward()
            optimizer.step()

            # Print log
            if verbose:
                logger.debug('[epoch:{}][Batch:{}/{}][loss:{:.4f}]'.format(epoch+1, i+1, n_batch, loss.item()))

            # Evaluate
            if cur_iter % evaluate_interval == 0:
                mAP = evaluate(model,
                               query_dataloader,
                               retrieval_dataloader,
                               code_length,
                               device,
                               topk,
                               multi_labels,
                               )
                logger.info('[iteration:{}][map:{:.4f}]'.format(cur_iter, mAP))

            # Save snapshot
            if cur_iter % snapshot_interval == snapshot_interval - 1:
                model.snapshot(cur_iter, optimizer)
                logger.info('[iteration:{}][Snapshot]'.format(cur_iter))

    # Evaluate and save snapshot
    mAP = evaluate(model,
                   query_dataloader,
                   retrieval_dataloader,
                   code_length,
                   device,
                   topk,
                   multi_labels,
                   )
    model.snapshot(cur_iter, optimizer)
    logger.info('Training finish, [iteration:{}][map:{:.4f}][Snapshot]'.format(cur_iter, mAP))


def evaluate(model, query_dataloader, retrieval_dataloader, code_length, device, topk, multi_labels):
    """
    Evaluate.

    Args
        model(torch.nn.Module): CNN model.
        query_dataloader(torch.evaluate.data.DataLoader): Query data loader.
        retrieval_dataloader(torch.evaluate.data.DataLoader): Retrieval data loader.
        code_length(int): Hash code length.
        device(torch.device): GPU or CPU.
        topk(int): Calculate top k data points map.
        multi_labels(bool): Multi labels.

    Returns
        mAP(float): Mean average precision.
    """
    model.eval()

    # Generate hash code
    query_code = generate_code(model, query_dataloader, code_length, device)
    retrieval_code = generate_code(model, retrieval_dataloader, code_length, device)

    # One-hot encode targets
    if multi_labels:
        onehot_query_targets = query_dataloader.dataset.get_targets().to(device)
        onehot_retrieval_targets = retrieval_dataloader.dataset.get_targets().to(device)
    else:
        onehot_query_targets = query_dataloader.dataset.get_onehot_targets().to(device)
        onehot_retrieval_targets = retrieval_dataloader.dataset.get_onehot_targets().to(device)

    # Calculate mean average precision
    mAP = mean_average_precision(
        query_code,
        retrieval_code,
        onehot_query_targets,
        onehot_retrieval_targets,
        device,
        topk,
    )

    model.train()

    return mAP


def generate_code(model, dataloader, code_length, device):
    """
    Generate hash code.

    Args
        model(torch.nn.Module): CNN model.
        dataloader(torch.evaluate.data.DataLoader): Data loader.
        code_length(int): Hash code length.
        device(torch.device): GPU or CPU.

    Returns
        code(torch.Tensor): Hash code.
    """
    with torch.no_grad():
        N = len(dataloader.dataset)
        code = torch.zeros([N, code_length])
        for data, _, index in dataloader:
            data = data.to(device)
            outputs = model(data)
            code[index, :] = outputs.sign().cpu()

    return code


def generate_similarity_matrix(features, alpha, beta):
    """
    Generate similarity matrix.

    Args
        features(torch.Tensor): Features.
        alpha, beta(float): Hyper-parameters.

    Returns
        S(torch.Tensor): Similarity matrix.
    """
    # Cosine similarity
    cos_dist = squareform(pdist(features.numpy(), 'cosine'))

    # Find maximum count of cosine distance
    max_cnt, max_cos = 0, 0
    interval = 1. / 100
    cur = 0
    for i in range(100):
        cur_cnt = np.sum((cos_dist > cur) & (cos_dist < cur + interval))
        if max_cnt < cur_cnt:
            max_cnt = cur_cnt
            max_cos = cur
        cur += interval

    # Split features into two parts
    flat_cos_dist = cos_dist.reshape((-1, 1))
    left = flat_cos_dist[np.where(flat_cos_dist <= max_cos)[0]]
    right = flat_cos_dist[np.where(flat_cos_dist > max_cos)[0]]

    # Reconstruct gaussian distribution
    left = np.concatenate([left, 2 * max_cos - left])
    right = np.concatenate([2 * max_cos - right, right])

    # Model data using gaussian distribution
    left_mean, left_std = norm.fit(left)
    right_mean, right_std = norm.fit(right)

    # Construct similarity matrix
    S = (cos_dist < (left_mean - alpha * left_std)) * 1.0 + (cos_dist > (right_mean + beta * right_std)) * -1.0

    return torch.FloatTensor(S)


def extract_features(model, dataloader, num_features, device, verbose):
    """
    Extract features.

    Args
        model(torch.nn.Module): CNN model.
        dataloader(torch.evaluate.data.DataLoader): Data loader.
        num_features(int): Number of features.
        device(torch.device): Using GPU or CPU.
        verbose(bool): Print log.

    Returns
        features(torch.Tensor): Features.
    """
    model.eval()
    model.set_extract_features(True)
    features = torch.zeros(dataloader.dataset.data.shape[0], num_features)
    with torch.no_grad():
        N = len(dataloader)
        for i, (data, _, index) in enumerate(dataloader):
            if verbose:
                logger.debug('[Batch:{}/{}]'.format(i+1, N))
            data = data.to(device)
            features[index, :] = model(data).cpu()

    model.set_extract_features(False)
    model.train()

    return features


class SSDH_Loss(nn.Module):
    def __init__(self):
        super(SSDH_Loss, self).__init__()

    def forward(self, H, S):
        loss = (S.abs() * (H - S).pow(2)).sum() / (H.shape[0] ** 2)

        return loss
