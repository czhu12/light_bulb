import { connect } from 'react-redux';
import React from 'react';
import { zip } from 'lodash';
import Highlighter from "react-highlight-words";
import ReactDOM from 'react-dom';
import Select from 'react-select';

import { createJudgementsFromBatch, shortenText } from '../utils';
import { CLASSIFICATION_COLORS } from '../constants';
import { submitBatchJudgements } from '../actions';

class TextTaskBatchView extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      checkedItems: props.batchItems.items.map(() => true),
      hiddenItems: props.batchItems.items.map(() => true),
      selectedLabel: null,
    };
  }

  componentWillReceiveProps(nextProps) {
    let checkedItems = nextProps.batchItems.items.map(() => true);
    let hiddenItems = nextProps.batchItems.items.map(() => true);
    let selectedLabel = this.state.selectedLabel;
    if (!!!selectedLabel) {
      selectedLabel = { value: 0, label: nextProps.task.classes[0] };
    }
    this.setState({checkedItems, hiddenItems, selectedLabel: selectedLabel});
  }

  checkedBox(idx, e) {
    let target = e.target;
    let newCheckedItems = this.state.checkedItems.slice(0);
    newCheckedItems[idx] = target.checked;
    this.setState({checkedItems: newCheckedItems});
  }

  _expandText(idx) {
    let newHiddenItems = this.state.hiddenItems.slice(0);
    newHiddenItems[idx] = false;
    this.setState({hiddenItems: newHiddenItems})
  }

  _collapseText(idx) {
    let newHiddenItems = this.state.hiddenItems.slice(0);
    newHiddenItems[idx] = true;
    this.setState({hiddenItems: newHiddenItems})
  }

  _handleChangeLabel(selectedLabel) {
    this.setState({ selectedLabel })
  }

  render() {
    let items = this.props.batchItems.items;
    let fetchingView = this.props.batchItems.fetching ?
      (<i className="fa fa-spinner fa-spin" aria-hidden="true"></i>) :
        (<i className="fa fa-arrow-right" aria-hidden="true"></i>);
    const options = this.props.task.classes.map((cls, idx) => {
      return {value: idx, label: cls};
    });

    let labelDropdown = (
			<div className="form-group">
				<label>Label as:</label>
        <Select
          value={this.state.selectedLabel}
          onChange={this._handleChangeLabel.bind(this)}
          options={options}
        />
			</div>

    );

    let batchItems = items.map((item, idx) => {
      let text = null;
      let expandCollapseButton = null;
      if (this.state.hiddenItems[idx]) {
        text = shortenText(item['text'], 50);
        expandCollapseButton = text !== item['text'] ? (
          <div style={{cursor: 'pointer'}} onClick={this._expandText.bind(this, idx)}>
            <b>Expand <i className="fas fa-chevron-down"></i></b>
          </div>
        ) : null;
      } else {
        text = item['text'];
        expandCollapseButton = (
          <div style={{cursor: 'pointer'}} onClick={this._collapseText.bind(this, idx)}>
            <b>Collapse <i className="fas fa-chevron-up"></i></b>
          </div>
        );
      }
      return (
        <div>
          <div className="row">
            <div className="col-sm-1">
              <input
                className="form-check-input"
                type="checkbox"
                checked={this.state.checkedItems[idx]}
                onChange={this.checkedBox.bind(this, idx)}
              />
            </div>
            <div
              className="col-sm-10"
            >
							<Highlighter
								highlightClassName="highlighted-search-query"
								caseSensitive={false}
								searchWords={[this.props.batchItems.searchQuery]}
								autoEscape={true}
								textToHighlight={text}
							/>
              {expandCollapseButton}
            </div>
          </div>
          <hr/>
        </div>
      );
    });

    const numSelected = this.state.checkedItems.filter((c) => c).length;
    let batchLabelView = null;
    if (this.props.batchItems.items.length > 0) {
      batchLabelView = (
        <div>
          <div style={{'textAlign': 'left'}}>
          {labelDropdown}
          </div>

          {batchItems}
          <a
            onClick={this.props.onClickSubmit.bind(
              null,
              this.props.batchItems.items,
              this.state.checkedItems,
              this.state.selectedLabel,
              this.props.task.classes,
            )}
            className="btn btn-lg">
            Label ({numSelected}) as <b>{this.props.task.classes[this.state.selectedLabel ? this.state.selectedLabel.value : 0]}</b> {fetchingView}
          </a>
        </div>
      );
    } else {
      batchLabelView = (
        <div className="mt-5">
          <h4>No unlabelled results found for "{this.props.batchItems.searchQuery}"</h4>
        </div>
      );
    }
    return batchLabelView;
  }
}

const mapStateToProps = state => ({
  task: state.task,
  batchItems: state.batchItems,
});

const mapDispatchToProps = dispatch => ({
  onClickSubmit: (items, checkedItems, label, classes) => {
    const judgements = createJudgementsFromBatch(items, checkedItems, label.value, classes);
    dispatch(submitBatchJudgements(judgements, true));
  }
});

export default connect(
  mapStateToProps,
  mapDispatchToProps,
)(TextTaskBatchView);
