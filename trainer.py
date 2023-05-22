import random
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import trange

from datasets import *
torch.set_printoptions(profile="full")

logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def get_input_from_batch(batch):
    inputs = { 'word_ids':batch[0],
               'wType_ids':batch[1],
                }
    labels = batch[2]

    return inputs, labels


def get_collate_fn():
    return my_collate

def train(args,model,train_dataset,test_dataset,train_labels_weight,test_labels_weight):
    '''Train the model'''
    tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size
    train_sampler = RandomSampler(train_dataset)
    collate_fn = get_collate_fn()
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                  batch_size=args.train_batch_size,
                                  collate_fn=collate_fn)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (
            len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(
            train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs


    parameters = filter(lambda param: param.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=args.learning_rate)

    # Train
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d",
                args.per_gpu_train_batch_size)
    logger.info("  Gradient Accumulation steps = %d",
                args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    all_ef_results = []
    all_er_results = []
    all_eu_results = []
    all_eo_results = []
    all_ep_results = []
    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    set_seed(args)
    epoch = 0

    train_labels_weight = torch.tensor(train_labels_weight).to(args.device)
    test_labels_weight = torch.tensor(test_labels_weight).to(args.device)

    f = open('./output/result.txt','w',encoding='utf-8')
    for _ in train_iterator:
        for step, batch in enumerate(train_dataloader):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs, labels = get_input_from_batch(batch)
            logits = model(**inputs)

            loss = F.cross_entropy(logits,labels,weight=train_labels_weight)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                # Log metrics
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    tb_writer.add_scalar(
                        'train_loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logger.info("  train_loss: %s", str((tr_loss - logging_loss) / args.logging_steps))
                    logging_loss = tr_loss

        results,eval_loss = evaluate(args,test_dataset,model,test_labels_weight,f)
        all_ef_results.append(results["EquityFreeze"])
        all_er_results.append(results["EquityRepurchase"])
        all_eu_results.append(results["EquityUnderweight"])
        all_eo_results.append(results["EquityOverweight"])
        all_ep_results.append(results["EquityPledge"])
        tb_writer.add_scalar('train_epoch_loss',(tr_loss - logging_loss) / args.logging_steps, epoch)
        epoch += 1

    tb_writer.close()
    return global_step, all_ef_results, all_er_results, all_eu_results, all_eo_results, all_ep_results

def evaluate(args, eval_dataset, model,test_labels_weight,f):
    args.eval_batch_size = args.per_gpu_eval_batch_size
    eval_sampler = SequentialSampler(eval_dataset)
    collate_fn = get_collate_fn()
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,batch_size=args.eval_batch_size,collate_fn=collate_fn)
    # Eval
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    out_label_ids = []
    final_preds = []

    for batch in eval_dataloader:
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        inputs, labels = get_input_from_batch(batch)

        logits = model(**inputs)
        loss = F.cross_entropy(logits,labels,weight=test_labels_weight)

        # tmp_eval_loss = loss
        eval_loss += loss.mean().item()
        nb_eval_steps += 1

        preds = np.argmax(logits.detach().cpu().numpy(), axis=1)
        final_preds += preds.tolist()
        out_label_ids += labels.detach().cpu().tolist()

    # preds = np.argmax(preds, axis=1)

    eval_loss = eval_loss / nb_eval_steps
    result = compute_metrics(final_preds, out_label_ids,idx2role_role)

    logger.info('***** Eval results *****')
    logger.info(" eval loss: %s", str(eval_loss))
    logger.info("************ef*************")
    for event_type_idx,event_type in idx2event_type.items():
        logger.info("************%s*************",event_type)
        for key,value in result[event_type].items():
            logger.info("  %s = %s", key, str(value))
            f.write(key+'='+str(value)+'\n')

    return result,eval_loss

def compute_metrics(preds,labels,idx2etype_role_role):
    event_mat = {"EquityFreeze": {"TP": 0, "FP": 0, "TP_FN": 0}, "EquityRepurchase": {"TP": 0, "FP": 0, "TP_FN": 0},
              "EquityUnderweight": {"TP": 0, "FP": 0, "TP_FN": 0}, "EquityOverweight": {"TP": 0, "FP": 0, "TP_FN": 0},
              "EquityPledge": {"TP": 0, "FP": 0, "TP_FN": 0}}

    for idx,label in enumerate(labels):
        pred = preds[idx]
        if label <= 1:
            continue
        event_type_label_idx, _, _ = idx2etype_role_role[label]
        event_mat[event_type_label_idx]["TP_FN"] += 1
        if pred == label:
            event_mat[event_type_label_idx]["TP"] += 1
        else:
            event_mat[event_type_label_idx]["FP"] += 1

    result = {}
    for event_type,values in event_mat.items():
        pre = 0
        recall = 0
        f1 = 0
        if (values["TP"]+values["FP"]) != 0:
            pre = (values["TP"])/(values["TP"]+values["FP"])
        if values["TP_FN"] != 0:
            recall = values["TP"]/values["TP_FN"]
        if pre != 0 and recall != 0:
            f1 = 2*pre*recall/(pre+recall)
        result[event_type] = {'pre':pre,'recall':recall,'f1':f1,'TP':values["TP"],'FP':values["FP"],'TP_FN':values["TP_FN"]}

    return result
