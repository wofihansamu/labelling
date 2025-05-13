let mobilenetModel;
let model;
const imageContainer = document.getElementById('image-container');
const imageUpload = document.getElementById('image-upload');
const trainButton = document.getElementById('train-button');
const predictButton = document.getElementById('predict-button');
const saveButton = document.getElementById('save-button');
const loadButton = document.getElementById('load-button');
const loadFromServerButton = document.getElementById('load-fromServer-button');

const images = [];
const labels = [];
let labelSet = [];

async function loadMobilenet() {
  mobilenetModel = await mobilenet.load({ version: 1, alpha: 1.0 });
  console.log('Model MobileNet berhasil dimuat.');
}

imageUpload.addEventListener('change', (event) => {
  const files = event.target.files;
  for (let file of files) {
    const reader = new FileReader();
    reader.onload = (e) => {
      const img = new Image();
      img.src = e.target.result;
      img.onload = () => {
        imageContainer.appendChild(img);
        const label = prompt('Masukkan label untuk gambar ini:');
        images.push(img);
        labels.push(label);
      };
    };
    reader.readAsDataURL(file);
  }
});

trainButton.addEventListener('click', async () => {
  const embeddings = [];
  for (let img of images) {
    const input = preprocessImage(img);
    const activation = mobilenetModel.infer(input, true);
    embeddings.push(activation);
  }

  const xs = tf.concat(embeddings);
  labelSet = [...new Set(labels)];
  const labelIndices = labels.map(label => labelSet.indexOf(label));
  const labelTensor = tf.tensor1d(labelIndices, 'int32');
  const ys = tf.oneHot(labelTensor, labelSet.length);
  console.log('Konversi label ke one-hot encoding DONE')

  model = tf.sequential();
  model.add(tf.layers.dense({ inputShape: [embeddings[0].shape[1]], units: 100, activation: 'relu' }));
  model.add(tf.layers.dense({ units: labelSet.length, activation: 'softmax' }));
  model.compile({ loss: 'categoricalCrossentropy', optimizer: 'adam', metrics: ['accuracy'] });
  console.log('Model Iniated');

  await model.fit(xs, ys, {
    epochs: 10,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        console.log(`Epoch ${epoch + 1}: loss = ${logs.loss}, accuracy = ${logs.acc}`);
      }
    }
  });

  console.log('Pelatihan selesai.');
});

predictButton.addEventListener('click', async () => {
  const file = await selectImageFile();
  const img = new Image();
  img.src = URL.createObjectURL(file);
  img.onload = async () => {
    const input = preprocessImage(img);
    const activation = mobilenetModel.infer(input, true);
    const prediction = model.predict(activation);
    prediction.print();
    const predictedIndex = (await prediction.argMax(-1).data())[0];
    const predictedLabel = labelSet[predictedIndex];
    alert(`Prediksi: ${predictedLabel}`);
  };
});

saveButton.addEventListener('click', async () => {
    if (model) {
        try {
          const modelName = 'my-trained-model';
          await model.save('downloads://'+modelName);
          const blob = new Blob([JSON.stringify(labelSet)], { type: 'application/json' });
          const a = document.createElement('a');
          a.href = URL.createObjectURL(blob);
          a.download = modelName+'.label';
          a.click();
          alert('Model dan labelSet berhasil disimpan.');
        } catch (error) {
          console.error('Gagal menyimpan model:', error);
          alert('Terjadi kesalahan saat menyimpan model.');
        }
      } else {
        alert('Model belum dilatih.');
      }
  });

  loadFromServerButton.addEventListener('click', async () => {
    try {
      const response = await fetch('/models/my-trained-model.label');
      labelSet = await response.json();
      model = await tf.loadLayersModel('/models/my-trained-model.json');
      alert('Model berhasil dimuat.');
    } catch (error) {
      console.error('Gagal memuat model atau labelSet:', error);
      alert('Terjadi kesalahan saat memuat.');
    }
  });

  loadButton.addEventListener('click', async () => {
    try {
        const input = document.createElement('input');
        input.type = 'file';
        input.accept = '.json,.bin,.label';
        input.multiple = true;
        input.onchange = async () => {
          const files = Array.from(input.files);
          const reader = new FileReader();
          reader.onload = (e) => {
            labelSet = JSON.parse(e.target.result);
            console.log('LabelSet berhasil dimuat.');
          };
          const jsonFile = files.find(f => f.name.endsWith('.json'));
          const binFile = files.find(f => f.name.endsWith('.bin'));
          const labelFile = files.find(f => f.name.endsWith('.label'));
    
          if (!labelFile || !jsonFile || !binFile) {
            alert('Harap pilih file .label, .json dan .bin dari model.');
            return;
          }
          reader.readAsText(labelFile);
          model = await tf.loadLayersModel(tf.io.browserFiles([jsonFile, binFile]));
          alert('Model berhasil dimuat.');
        };
        input.click();
      } catch (error) {
        console.error('Gagal memuat model atau labelSet:', error);
        alert('Terjadi kesalahan saat memuat.');
      }
  });

function selectImageFile() {
  return new Promise((resolve) => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'image/*';
    input.onchange = () => {
      resolve(input.files[0]);
    };
    input.click();
  });
}

function preprocessImage(img) {
    return tf.tidy(() => {
      return tf.browser.fromPixels(img)
        .resizeNearestNeighbor([224, 224])
        .toFloat()
        .div(255.0)
        .expandDims();
    });
  }

loadMobilenet();
