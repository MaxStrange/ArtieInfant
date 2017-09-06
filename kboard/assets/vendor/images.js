var accChart;
var fscoreChart;
var lossChart;

function updateData(chart, y, metric) {
  if (chart.series != undefined) {
    var x = (new Date()).getTime(), // current time
    series = chart.series[0],
    shift = series.data.length > 20;
    series.addPoint([x, y], true, shift);
    setTimeout(function() {requestData(chart, metric)}, 2000);
  }
}

function requestData(chart, metric) {
  var x;
  if (chart.series != undefined) {
    x = chart.series[0].data.length;
  } else {
    x = 0;
  }
  $.ajax({
    url: "/data",
    data: {"x": x, "metric": metric},
    success: function(y) {
      updateData(chart, y, metric);
    },
    cache: false
  });
}

function makeChart(metric, title, container) {
  return new Highcharts.Chart({
    chart: {
      renderTo: container,
      defaultSeriesType: "spline",
      events: {
        load: function() {
          requestData(this, metric);
        }
      }
    },
    title: {
      text: title
    },
    xAxis: {
      type: "datetime",
      tickPixelInterval: 150,
      maxZoom: 20 * 1000
    },
    yAxis: {
      minPadding: 0.2,
      maxPadding: 0.2,
      title: {
        text: title,
        margin: 80
      }
    },
    series: [{
      name: title,
      data: []
    }]
  });
}

$(document).ready(function() {
  accChart = makeChart("acc", "Accuracy", "accContainer");
  fscoreChart = makeChart("fscore", "FScore", "fscoreContainer");
  lossChart = makeChart("loss", "Loss", "lossContainer");
});
