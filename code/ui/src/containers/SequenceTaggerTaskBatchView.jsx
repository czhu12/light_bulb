import { connect } from 'react-redux';
import { zip } from 'lodash';
import React from 'react';
import { submitBatchJudgements } from '../actions';

import SequenceTaggerTaskView from './SequenceTaggerTaskView';

class SequenceTaggerTaskBatchView extends React.Component {
  constructor(props) {
    super(props);
    this.state = {selected: props.batchItems.items.map(() => true)};
  }

  componentDidUpdate(prevProps, prevState, snapshot) {
    if (prevProps.batchItems.items !== this.props.batchItems.items) {
      // Scroll to top.
    }
  }

  componentWillReceiveProps(nextProps) {
    let selected = nextProps.batchItems.items.map(() => true);
    this.setState({selected: selected});
  }

  render() {
    let items = this.props.batchItems.items;
    let sequenceTaggerView = items.map((item) => {
      return (
        <div>
          <SequenceTaggerTaskView task={this.props.task} currentItem={item} />
          <hr />
        </div>
      )
    });
    return (
      <div>
        {sequenceTaggerView}
      </div>
    );
  }
}

const mapStateToProps = state => ({
  currentSelectedClass: state.items.currentSelectedClass,
  task: state.task,
  batchItems: state.batchItems,
});

const mapDispatchToProps = dispatch => ({
  onClickSubmit: (items, selected, targetClass) => {
    let judgements = items.map((item, idx) => {
      return {
        path: item['path'],
        is_target_class: selected[idx],
        target_class: targetClass,
      }
    });

    dispatch(submitBatchJudgements(judgements));
  }
});

export default connect(
  mapStateToProps,
  mapDispatchToProps
)(SequenceTaggerTaskBatchView);
