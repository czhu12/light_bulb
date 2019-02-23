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

export const shortenText = (text, numWords) => {
  let split = text.split(' ');
  let halfWords = Math.floor(numWords / 2);
  if (split.length > numWords) {
    return split.slice(0, halfWords).concat(['...']).concat(split.slice(split.length - numWords, split.length)).join(' ');
  }
  return text;
}
