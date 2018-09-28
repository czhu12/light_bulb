import { connect } from 'react-redux';
import { zip } from 'lodash';
import React from 'react';
import { submitBatchJudgements, updateBatchItemByIndex } from '../actions';

import SequenceTaggerTaskView from './SequenceTaggerTaskView';

class SequenceTaggerTaskBatchView extends React.Component {
  constructor(props) {
    super(props);
  }

  componentDidUpdate(prevProps, prevState, snapshot) {
    if (prevProps.batchItems.items !== this.props.batchItems.items) {
      // Scroll to top.
    }
  }

  changedLabels(index, label) {
    let item = this.props.batchItems.items.slice()[index];
    item['label'] = JSON.stringify(label);
    this.props.updateBatchItemByIndex(index, item);
  }

  _submitLabels() {
    // Submit labels in this.props.batchItems.items;
  }

	componentDidMount() {
    window.addEventListener("keypress", this._submitLabels.bind(this));
	}

	componentWillUnmount() {
    window.removeEventListener("keypress", this._submitLabels.bind(this));
	}

  render() {
    let items = this.props.batchItems.items;
    let sequenceTaggerView = items.map((item, index) => {
      return (
        <div>
          <SequenceTaggerTaskView
            isBatchMode={true}
            task={this.props.task}
            currentItem={item}
            changedLabels={this.changedLabels.bind(this, index)}
          />
          <hr />
        </div>
      )
    });
    return (
      <div>
        {sequenceTaggerView}
        <a
          onClick={this._submitLabels.bind(this)}
          className="btn btn-lg">
          Submit
        </a>
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
  updateBatchItemByIndex: (index, item) => {
    dispatch(updateBatchItemByIndex(index, item));
  }
});

export default connect(
  mapStateToProps,
  mapDispatchToProps
)(SequenceTaggerTaskBatchView);
