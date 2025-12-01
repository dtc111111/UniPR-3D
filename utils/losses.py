from training.train_utils.general import check_and_fix_inf_nan

def get_loss(loss_name):
    from pytorch_metric_learning import losses
    from pytorch_metric_learning.distances import DotProductSimilarity

    if loss_name == 'SupConLoss': return losses.SupConLoss(temperature=0.07)
    if loss_name == 'CircleLoss': return losses.CircleLoss(m=0.4, gamma=80) #these are params for image retrieval
    if loss_name == 'MultiSimilarityLoss': return losses.MultiSimilarityLoss(alpha=1.0, beta=50, base=0.0, distance=DotProductSimilarity())
    if loss_name == 'ContrastiveLoss': return losses.ContrastiveLoss(pos_margin=0, neg_margin=1)
    if loss_name == 'Lifted': return losses.GeneralizedLiftedStructureLoss(neg_margin=0, pos_margin=1, distance=DotProductSimilarity())
    if loss_name == 'FastAPLoss': return losses.FastAPLoss(num_bins=30)
    if loss_name == 'NTXentLoss': return losses.NTXentLoss(temperature=0.07) #The MoCo paper uses 0.07, while SimCLR uses 0.5.
    if loss_name == 'TripletMarginLoss': return losses.TripletMarginLoss(margin=0.1, swap=False, smooth_loss=False, triplets_per_anchor='all') #or an int, for example 100
    if loss_name == 'CentroidTripletLoss': return losses.CentroidTripletLoss(margin=0.05,
                                                                            swap=False,
                                                                            smooth_loss=False,
                                                                            triplets_per_anchor="all",)
    raise NotImplementedError(f'Sorry, <{loss_name}> loss function is not implemented!')

def get_miner(miner_name, margin=0.1):
    from pytorch_metric_learning import miners
    from pytorch_metric_learning.distances import CosineSimilarity, DotProductSimilarity

    if miner_name == 'TripletMarginMiner' : return miners.TripletMarginMiner(margin=margin, type_of_triplets="semihard") # all, hard, semihard, easy
    if miner_name == 'MultiSimilarityMiner' : return miners.MultiSimilarityMiner(epsilon=margin, distance=CosineSimilarity())
    if miner_name == 'PairMarginMiner' : return miners.PairMarginMiner(pos_margin=0.7, neg_margin=0.3, distance=DotProductSimilarity())
    return None




def compute_camera_loss_3dof(
    pred_pose_encodings,    
    gt_pose_encoding,       # ground truth and mask batch dict
    loss_type="l1",         # "l1" or "l2" loss
    gamma=0.6,              # temporal decay weight for multi-stage training
    weight_trans=1.0,       # weight for translation loss
    weight_rot=1.0,         # weight for rotation loss
    **kwargs
):
    # List of predicted pose encodings per stage
    # pred_pose_encodings = pred_dict['pose_enc_list']
    # Binary mask for valid points per frame (B, N, H, W)
    # Number of prediction stages
    n_stages = len(pred_pose_encodings)

    # gt_pose_encoding = batch_data['camera_pose_encodings']
    # Initialize loss accumulators for translation, rotation, focal length
    total_loss_T = total_loss_R = total_loss_FL = 0

    # Compute loss for each prediction stage with temporal weighting
    for stage_idx in range(n_stages):
        # Later stages get higher weight (gamma^0 = 1.0 for final stage)
        stage_weight = gamma ** (n_stages - stage_idx - 1)
        pred_pose_stage = pred_pose_encodings[stage_idx]


            # Only consider valid frames for loss computation
        loss_T_stage, loss_R_stage = camera_loss_single_3dof(
            pred_pose_stage.clone(),
            gt_pose_encoding.clone(),
            loss_type=loss_type
        )
        # Accumulate weighted losses across stages
        total_loss_T += loss_T_stage * stage_weight
        total_loss_R += loss_R_stage * stage_weight

    # Average over all stages
    avg_loss_T = total_loss_T / n_stages
    avg_loss_R = total_loss_R / n_stages

    # Compute total weighted camera loss
    total_camera_loss = (
        avg_loss_T * weight_trans +
        avg_loss_R * weight_rot
    )

    # Return loss dictionary with individual components
    return {
        "loss_camera": total_camera_loss,
        "loss_T": avg_loss_T,
        "loss_R": avg_loss_R
    }



def camera_loss_single_3dof(pred_pose_enc, gt_pose_enc, loss_type="l1"):
    """
    Computes translation, rotation, and focal loss for a batch of pose encodings.
    
    Args:
        pred_pose_enc: (N, D) predicted pose encoding
        gt_pose_enc: (N, D) ground truth pose encoding
        loss_type: "l1" (abs error) or "l2" (euclidean error)
    Returns:
        loss_T: translation loss (mean)
        loss_R: rotation loss (mean)
    
    NOTE: The paper uses smooth l1 loss, but we found l1 loss is more stable than smooth l1 and l2 loss.
        So here we use l1 loss.
    """
    if loss_type == "l1":
        # Translation: first 2 dims; Rotation: next 4 (quaternion); Focal/Intrinsics: last dims
        loss_T = (pred_pose_enc[..., :2] - gt_pose_enc[..., :2]).abs()
        loss_R = (pred_pose_enc[..., 2:3] - gt_pose_enc[..., 2:3]).abs()
    elif loss_type == "l2":
        # L2 norm for each component
        loss_T = (pred_pose_enc[..., :2] - gt_pose_enc[..., :2]).norm(dim=-1, keepdim=True)
        loss_R = (pred_pose_enc[..., 2:3] - gt_pose_enc[..., 2:3]).norm(dim=-1)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    # Check/fix numerical issues (nan/inf) for each loss component
    loss_T = check_and_fix_inf_nan(loss_T, "loss_T")
    loss_R = check_and_fix_inf_nan(loss_R, "loss_R")

    # Clamp outlier translation loss to prevent instability, then average
    loss_T = loss_T.clamp(max=100).mean()
    loss_R = loss_R.mean()

    return loss_T, loss_R