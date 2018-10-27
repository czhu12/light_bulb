import { chunk, zip } from 'lodash';
import { connect } from 'react-redux';
import React from 'react';

class DatasetApp extends React.Component {
  _cardView(item) {
    let text = item[0].text;
    let path = item[0].path;
    let file = path.split('/')[path.split('/').length - 1];
    let entropy = Math.round(item[1] * 100) / 100;
    let prediction = item[3];
		return (
			<div className="card card-dataset" style={{width: "85%", height: "250px"}}>
        <div className="card-header">{file}</div>
				<div className="card-body">
					<h5 className="card-title">{text}</h5>
					<div>{prediction}</div>

				</div>
        <div className="card-footer">
          <p class="card-text">Entropy: {entropy}</p>
        </div>
			</div>
		);
  }

  render() {
    let items = zip(this.props.items, this.props.entropy, this.props.stages, this.props.predictions);
    items.sort((a, b) => {
      return b[1] - a[1];
    });
    let itemRows = chunk(items, 4).map((rowItems) => {
      return rowItems.map((item) => {
        return (
					<div className="col-md-3">
						{this._cardView(item)}
					</div>
				);
      });
    }).map((row) => {
      return <div className="row">{row}</div>;
    });
    return (
      <div>
        {itemRows}
      </div>
    );
  }
}

const mapStateToProps = state => ({
  items: state.items.items,
  entropy: state.items.entropy,
  stages: state.items.stages,
  predictions: state.items.predictions,
});


const mapDispatchToProps = dispatch => ({
});

export default connect(
  mapStateToProps,
  mapDispatchToProps,
)(DatasetApp);
