import { zip } from 'lodash';
import { connect } from 'react-redux';
import React from 'react';
import NavigationBar from './NavigationBar';
import BoundingBoxDemoView from './BoundingBoxDemoView';
import { submitDataToScore, changeDemoScoreUrlSequence, changeDemoScoreText } from '../actions';

class DemoApp extends React.Component {
  render() {
    let demoView = null;
    let firstPrediction = null;
    let predictionClasses = null;
    if (this.props.demo.predictions.length > 0) {
      firstPrediction = this.props.demo.predictions[0];
      let argMaxIndex = firstPrediction.map((x, i) => [x, i]).reduce((r, a) => (a[0] > r[0] ? a : r))[1];
      // Argmax from https://gist.github.com/engelen/fbce4476c9e68c52ff7e5c2da5c24a28
      let classesAndScores = zip(this.props.task.classes, firstPrediction)
      predictionClasses = classesAndScores.map((classAndScore, index) => {
        let cls = classAndScore[0];
        let score = classAndScore[1];
        score = Math.round(score * 100)
        if (index === argMaxIndex) {
          return (<b><span style={{paddingRight: "5px"}}>{cls} ({score}%)</span></b>);
        } else {
          return (<span style={{paddingRight: "5px"}}>{cls} ({score}%)</span>);
        }
      });
    }

    if (this.props.task.dataType === 'images') {
      demoView = (
				<div>
          <form>
            <div className="form-group">
              <label>
                {this.props.task.title}
              </label>

              <input
                value={this.props.demo.urlSequence}
                placeholder="Image URL"
                className="form-control"
                rows="3"
                onChange={this.props.onChangeUrlSequence}
              />
            </div>
            <button
              onClick={this.props.submitDataToScore}
              className="btn btn-primary"
            >
              Classify Image
            </button>
          </form>
          <div className="row">
            <div className="col-md-6 offset-md-2">
              <img className="img-fluid" src={this.props.demo.url} />
            </div>
          </div>
				</div>
      )
    } else if (this.props.task.dataType === 'text') {
      demoView = (
        <div>
          <form>
            <div className="form-group">
              <label>
                {this.props.task.title}
              </label>
              <textarea
                value={this.props.demo.text}
                className="form-control"
                rows="3"
                onChange={this.props.onChangeText}
              ></textarea>
            </div>
            <button
              onClick={this.props.submitDataToScore}
              className="btn btn-primary">
              Classify Text
            </button>
          </form>
        </div>
      )
    } else if (this.props.task.dataType == 'object_detection') {
      <BoundingBoxDemoView
        url={this.props.demo.url}
        boundingBoxes={firstPrediction}
      />
    }
    return (
      <div>
        <NavigationBar />
        <div className="container">
          { demoView }
          {predictionClasses}
        </div>
      </div>
    );
  }
}

const mapStateToProps = state => ({
  task: state.task,
  demo: state.demo,
});

const mapDispatchToProps = dispatch => ({
  submitDataToScore: (e) => {
    e.preventDefault();
    dispatch(submitDataToScore())
  },
  onChangeText: (e) => {
    let text = e.target.value;
    dispatch(changeDemoScoreText(text));
  },
  onChangeUrlSequence: (e) => {
    let urlSequence = e.target.value;
    dispatch(changeDemoScoreUrlSequence(urlSequence));
  },
});

export default connect(
  mapStateToProps,
  mapDispatchToProps,
)(DemoApp);
