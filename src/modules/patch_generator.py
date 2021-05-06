import numpy as np


def get_data_patch_stats(dataLen, sampleLen, config):
    """Computes the number of patches per config and the patches per sample.

    Args:
        dataLen (int): dataset length
        sampleLen (int): sample length
        config (list): stride, len pair for patches

    Returns:
        objects: number of patches per config, patches per sample
    """
    num_per_config = np.zeros(len(config))
    patch_per_sample = np.zeros(len(config))
    for i in range(len(config)):
        patch_per_sample[i] = int(np.ceil(sampleLen / config[i][0]))
        num_per_config[i] = int(patch_per_sample[i] * dataLen)
    return num_per_config, patch_per_sample


def get_patch_params(idx, dataLen, sampleLen, config):
    """Returns the parameters of the patch including sample id, start and end

    Args:
        idx (int): patch id
        dataLen (int): data length
        sampleLen (int): sample length
        config (int): stride, len pair for patches

    Returns:
        objects: sample id of the patch, start and end time-step
    """
    for c in config:
        patch_per_sample = int(np.ceil(sampleLen / c[0]))
        configLen = patch_per_sample * dataLen
        if not idx < configLen:
            idx -= configLen
        else:
            sidx = idx // patch_per_sample
            patch = idx % patch_per_sample
            start = patch * c[0]
            end = np.min([sampleLen, start+c[1]])
            return sidx, start, end


def get_patch_params_list(ids, dataLen, seqlen, config):
    """Computes the parameters for a lsit of ids.

    Args:
        ids (arr): array holding the ids for the parameter list
        dataLen (int): data length
        seqlen (int): sample length
        config (int): stride, len pair for patches

    Returns:
        arr: parameter array for the given patch ids
    """
    param_list = np.zeros((len(ids), 3), dtype=int)
    for c, i in enumerate(ids):
        sidx, start, end = get_patch_params(i, dataLen, seqlen, config)
        param_list[c] = np.array([sidx, start, end])
    return param_list


def get_patch(data, start, end, zero=False, attach=False, notemp=False):
    """Returns the patch after transformation.

    Args:
        data (arr): sample
        start (int): start time-step of the patch
        end (int): ent time-step of the patch
        zero (bool, optional): flag to zero the data outside of the patch. Defaults to False.
        attach (bool, optional): flag to attack a channel for the patch boundaries. Defaults to False.
        notemp (bool, optional): flag to remove time axis. Defaults to False.

    Returns:
        [type]: [description]
    """
    if not zero:
        result = np.copy(data)
    else:
        result = np.zeros(data.shape)
        if notemp:
            result[0:end-start] = data[start:end]
        else:
            result[start:end] = data[start:end]
    if attach:
        att = np.zeros((data.shape[0], 1))
        if notemp:
            att[0:end-start] = 1
        else:
            att[start:end] = 1
        result = np.concatenate([result, att], axis=-1)
    return result


def get_all_patch(ids, data, dataLen, seqlen, config, zero=False, attach=False, notemp=False):
    """Returns all patches of the given ids.

    Args:
        ids (arr): array holding the ids of the patches
        data (arr): dataset array
        dataLen (int): dataset length
        seqlen (int): sequence length
        config (list): stride, len pairs for the patches
        zero (bool, optional): flag to zero out data not included in the patch. Defaults to False.
        attach (bool, optional): flag to attach the boundary channel. Defaults to False.
        notemp (bool, optional): flag to remove the time axis. Defaults to False.

    Returns:
        arr: array hlding the patches.
    """
    samples = []
    for i in ids:
        sidx, start, end = get_patch_params(i, dataLen, seqlen, config)
        sample_complete = get_patch(
            data[sidx], start, end, zero, attach, notemp)
        samples.append(sample_complete)
    samples = np.asarray(samples)
    return samples


def get_all_patch_params(idx, num_per_config, patch_per_sample):
    """Computes the ids of all patches for a given id.

    Args:
        idx (int): id of the sample
        num_per_config (arr): number of patches per config
        patch_per_sample (arr): numbper of patches per sample

    Returns:
        arr: ids that hold patches of the given sample
    """
    ids = []
    for i in range(len(num_per_config)):
        if i > 0:
            idx += num_per_config[i-1]
        ids.append(
            np.arange(idx*patch_per_sample[i], (idx+1)*patch_per_sample[i], dtype=int))
    ids = np.concatenate(ids)
    return ids


def get_generator_id_list(datalen, seqlen, config):
    """Returns the id list for the patches

    Args:
        datalen (int): length of the dataset
        seqlen (int): length of the sequences
        config (list): stride, len paits for the patches

    Returns:
        array: id list for each sample
    """
    npc, _ = get_data_patch_stats(datalen, seqlen, config)
    return np.arange(np.sum(npc), dtype=int)


def get_sample_id_list(seqlen, patch_per_sample):
    """Returns the id list that provides the sample id for each patch

    Args:
        seqlen (int): lenth of the sample
        patch_per_sample (int): patches per sample

    Returns:
        [type]: [description]
    """
    result = []
    for i in range(len(patch_per_sample)):
        result.append(np.repeat(np.arange(seqlen), patch_per_sample[i]))
    result = np.concatenate(result)
    return result


def create_histo_dataset(softmax_preds, sidx, thresh=0.0, full=False, binary=False, most=False):
    """Creates the history for the level 2 classifier. Depending on parameter includes the full set of preds, or a filterd one.

    Args:
        softmax_preds (arr): softmax preds for level 1
        sidx (arr): sample ids
        thresh (float, optional): confidence thresh for indivitual pred to be included. Defaults to 0.0.
        full (bool, optional): extracts the metadata time-step wise. Defaults to False.
        binary (bool, optional): binary extraction. Defaults to False.
        most (bool, optional): extraction based on most occurances. Defaults to False.

    Returns:
        arr: histogram data for levle 2 processing
    """
    per_sample = np.unique(sidx, return_counts=True)[1][0]
    if full:
        patch_off_counts = np.zeros(
            int(softmax_preds.shape[0] / per_sample), dtype=int)
        dataset = np.zeros(
            (int(softmax_preds.shape[0] / per_sample), int(per_sample * softmax_preds.shape[1])))
    else:
        dataset = np.zeros(
            (int(softmax_preds.shape[0] / per_sample), softmax_preds.shape[1]))
    for s in range(softmax_preds.shape[0]):
        idx = sidx[s]
        if binary:
            vals = np.array([1 if v > thresh else 0 for v in softmax_preds[s]])
        elif most:
            vals = np.zeros(softmax_preds.shape[1])
            maxarg = np.argmax(softmax_preds[s])
            vals[maxarg] += 1
        else:
            vals = np.array([v if v > thresh else 0 for v in softmax_preds[s]])

        if full:
            patch_off = patch_off_counts[idx]

            dataset[idx, patch_off:patch_off+len(vals)] = vals
            patch_off_counts[idx] += len(vals)
        else:
            dataset[idx] += vals
    return dataset


def compute_trivial_preds(softmax_preds, sids, mode='majority', occurance=1):
    """Compute the level 2 prediction using trivial approach

    Args:
        softmax_preds (arr): softmax predictions of level 1
        sids (arr): sample ids
        mode (str, optional): mode can be either majority or occurance. Defaults to 'majority'.
        occurance (int, optional): number of mimimum occurances. Defaults to 1.

    Returns:
        arr: prediction array for level 2
    """
    preds_arg = create_histo_dataset(
        softmax_preds, sids, full=False, most=True)
    if mode == 'majority':
        return np.argmax(preds_arg, axis=-1)
    else:
        return np.array([occurance if d[occurance] > 0 else 1 - occurance for d in preds_arg])