import { connect } from 'react-redux';
import React from 'react';

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

  onClickImage(idx) {
    let selected = this.state.selected.slice(0);
    selected[idx] = !selected[idx];
    this.setState({selected: selected});
  }

  render() {
    let items = this.props.batchItems.items;

    return (
      <div>
        {images}
        <div className="pull-right">
          <a
            onClick={this.props.onClickSubmit.bind(
              null,
              this.props.batchItems.items,
              this.state.selected,
              this.props.batchItems.targetClass,
            )}
            style={{color: '#eee', backgroundColor: color}}
            className="btn btn-lg">
            Submit {fetchingView}
          </a>
        </div>
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
