export const createJudgementsFromBatch = (items, selected, targetClass, classes) => {
  let judgements = items.map((item, idx) => {
    item['is_target_class'] = selected[idx];
    return item
  }).filter((item, idx) => {
    if (classes.length === 2) {
        // If there are only two classes, then we can infer the class
        // even if its not selected.
      return true
    } else {
      return selected[idx];
    }
  }).map((item, idx) => {
    if (classes.length !== 2 || selected[idx]) {
      item['label'] = targetClass;
    } else {
      // If there are only two classes, then we can infer the class
      // even if its not selected.
      item['label'] = targetClass === 1 ? 0 : 1;
    }
    return item;
  });

  return judgements;
};
