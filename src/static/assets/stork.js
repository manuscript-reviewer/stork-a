// Env variables
const maxImagesPermitted = 20;
const baseApiUrl = 'api'
// End Env variables

let token = getCookie('stork-auth');
checkTokenAndRedirectIfEmpty();

setInterval(function(){
    checkTokenAndRedirectIfEmpty();
}, 5000);

function checkTokenAndRedirectIfEmpty() {
    token = getCookie('stork-auth');
    if (window.location.pathname !== '/login' && token === '') {
        window.location.href = '/login';
    }
}

function login(event) {
    postLoginData(event.srcElement.username.value, event.srcElement.password.value, baseApiUrl);
}

function postLoginData(username, password, baseApiUrl) {
    let xhr = new XMLHttpRequest();
    xhr.open('POST', `${baseApiUrl}/login`, true);
    let formData = new FormData();
    formData.append('username', username);
    formData.append('password', password);

    xhr.onload = (e) => {
        if (xhr.status === 200) {
            window.location.href = '/';
        } else {
            document.getElementById('login-form-username-password-error').classList.remove('hidden');
        }
    };
    xhr.onerror = (e) => alert('An error occurred!');
    xhr.send(formData);
}

const loginForm = document.getElementById('login-form');
if (loginForm) {
    loginForm.addEventListener('submit', (event) => {
        event.preventDefault();
        login(event);
    })
}

function getCookie(cookieName) {
    var name = cookieName + '=';
    var decodedCookie = decodeURIComponent(document.cookie);
    var ca = decodedCookie.split(';');
    for(var i = 0; i <ca.length; i++) {
        var c = ca[i];
        while (c.charAt(0) == ' ') {
            c = c.substring(1);
        }
        if (c.indexOf(name) == 0) {
            return c.substring(name.length, c.length);
        }
    }
    return '';
}

function average(array) {
    return array.reduce((a,b) => a + b, 0) / array.length;
}

function isAnImage(file) {
    if (file && file.type) {
        return file.type.startsWith('image/jpeg') || file.type.startsWith('image/png') || file.type.startsWith('image/tiff');
    }

    return false;
};

function getFormData(file, data) {
    const formData = new FormData();
    if (isAnImage(file)) {
        formData.append('image', file, file.name);
    }
    formData.append('data', JSON.stringify(data));

    return formData;
};

function postFormData(formData, baseApiUrl, fileName) {
    let xhr = new XMLHttpRequest();
    xhr.open('POST', `${baseApiUrl}/upload`, true);
    xhr.setRequestHeader('Authorization','Basic ' + token);

    const card = document.getElementById(`image-card-${fileName}`);
    const submitBtn = document.getElementById(`submit-btn-${fileName}`);
    const loaders = [...card.getElementsByClassName('loader')];
    loaders.map(x => x.classList.remove('hidden'));

    xhr.onload = function (e) {
        if (xhr.status === 200) {
            response = JSON.parse(e.target.response);
            currentImages[fileName].result = response;

            const results = card.getElementsByClassName('results');
            showResultData(results[0], response.abnormalNormal);
            showResultData(results[1], response.cxAEUP);
            showResultData(results[2], response.cxAEverything);

            submitBtn.parentElement.classList.add('hidden');
        } else {
            alert('An error occurred!');
            submitBtn.removeAttribute('disabled');
            [...card.getElementsByTagName('select')].forEach(x => x.removeAttribute('disabled'));
            [...card.getElementsByTagName('input')].forEach(x => x.removeAttribute('disabled'));
        }
        loaders.map(x => x.classList.add('hidden'));
    };
    xhr.onerror = () => {
      loaders.map(x => x.classList.add('hidden'));
    };

    xhr.send(formData);
};

function showResultData(resultElement, data) {
    const bar = resultElement.getElementsByClassName('bar')[0];
    const goodText = resultElement.getElementsByClassName('good-text')[0];
    const poorText = resultElement.getElementsByClassName('poor-text')[0];
    const goodPercentage = (data.confidence[1] * 100).toFixed(2);
    const poorPercentage = (data.confidence[0] * 100).toFixed(2);
    bar.setAttribute('style', `width:${goodPercentage}%;`);
    goodText.innerHTML = `${goodPercentage}%`;
    poorText.innerHTML = `${poorPercentage}%`;
    if (goodPercentage > poorPercentage) {
      resultElement.getElementsByClassName('good-result-text')[0].classList.remove('hidden')
    } else {
      resultElement.getElementsByClassName('poor-result-text')[0].classList.remove('hidden')
    }
    resultElement.classList.remove('hidden');
}

function removeImageCard(imageName) {
    const card = document.getElementById(`image-card-${imageName}`);
    imagesPlaceholder.removeChild(card);
    delete currentImages[imageName];
    imageRemovedUpdateUI();
};

function imageRemovedUpdateUI() {
    if (Object.keys(currentImages).length === 0) {
        clearAllButton.classList.add('disabled');
    }
};

function clearAllImageCards() {
    Object.keys(currentImages).map(image => {
        removeImageCard(image)
    });
};

function createImageUIFromFile(file, imagesPlaceholder) {
    if (file === null || file === undefined || imagesPlaceholder === null || imagesPlaceholder === undefined) {
        return;
    }

    const imagePlaceholder = document.createElement('div');
    imagePlaceholder.classList.add('image-card');
    imagePlaceholder.id = `image-card-${file.name}`;
    imagePlaceholder.innerHTML = `
        <div class="card image-container">
            <div class="card-image">
                <img alt="${file.name}" width="330px" />
            </div>
            <div class="card-file-name">${file.name}</div>

            <div class="loader hidden"></div>
            <div class="delete-image-button" onclick="removeImageCard('${file.name}')">
                <i class="material-icons">clear</i>
            </div>

            <div class="card-content">
                <div class="results hidden">
                    <div class="result-text">Abnormal/Normal:
                        <strong class="good-result-text hidden">Euploid</strong>
                        <strong class="poor-result-text hidden">Aneuploid</strong>
                    </div>
                    <div class="poor">
                        <div class="good bar"></div>
                    </div>
                    <div class="legend-item"><div class="legend-marker good"></div>Euploid: <span class="good-text"></span></div>
                    <div class="legend-item"><div class="legend-marker poor"></div>Aneuploid: <span class="poor-text"></span></div>
                </div>


                <div class="results  hidden">
                    <div class="result-text">CxA-EUP:
                        <strong class="good-result-text hidden">Euploid</strong>
                        <strong class="poor-result-text hidden">Complex Aneuploid</strong>
                    </div>
                    <div class="poor">
                        <div class="good bar"></div>
                    </div>
                    <div class="legend-item"><div class="legend-marker good"></div>Euploid: <span class="good-text"></span></div>
                    <div class="legend-item"><div class="legend-marker poor"></div>Complex Aneuploid: <span class="poor-text"></span></div>
                </div>


                <div class="results  hidden">
                    <div class="result-text">CxA-Everything:
                        <strong class="good-result-text hidden">Not Complex Aneuploid</strong>
                        <strong class="poor-result-text hidden">Complex Aneuploid</strong>
                    </div>
                    <div class="poor">
                        <div class="good bar"></div>
                    </div>
                    <div class="legend-item"><div class="legend-marker good"></div>Not Complex Aneuploid: <span class="good-text"></span></div>
                    <div class="legend-item"><div class="legend-marker poor"></div>Complex Aneuploid: <span class="poor-text"></span></div>
                </div>
            </div>

            <div class="card-content">
                <div class="input-field">
                    <select class="egg-age-select">
                        <option value selected> -- select an option -- </option>
                        ${range(21, 48).map(x => `<option value="${x}">${x}</option>`)}
                    </select>
                    <label>Age</label>
                </div>

                <div class="input-field">
                    <select class="blastocyst-score-select">
                        <option value selected> -- select an option -- </option>
                        ${blastocystScoreOptions.map(x => `<option value="${x}">${x}</option>`)}
                    </select>
                    <label>Blastocyst Score (BS)</label>
                </div>

                <ul class="collapsible">
                    <li>
                        <div class="collapsible-header">Blastocyst Grade (BG)</div>
                        <div class="collapsible-body">
                            <div class="input-field">
                                <select class="blastocyst-grade-expansion-select">
                                    <option value selected> -- select an option -- </option>
                                    ${blastocystGradeOptions.expansionOptions.map(x => `<option value="${x}">${x}</option>`)}
                                </select>
                                <label>Expansion</label>
                            </div>
                            <div class="input-field">
                                <select class="blastocyst-grade-inner-cell-mass-select">
                                    <option value selected> -- select an option -- </option>
                                    ${blastocystGradeOptions.innerCellMassOptions.map(x => `<option value="${x}">${x}</option>`)}
                                </select>
                                <label>Inner Cell Mass (ICM)</label>
                            </div>
                            <div class="input-field">
                                <select class="blastocyst-grade-trophectoderm-select">
                                    <option value selected> -- select an option -- </option>
                                    ${blastocystGradeOptions.trophectodermOptions.map(x => `<option value="${x}">${x}</option>`)}
                                </select>
                                <label>Trophectoderm (TE)</label>
                            </div>
                        </div>
                    </li>
                </ul>

                <ul class="collapsible">
                    <li>
                        <div class="collapsible-header">Morphokinetics</div>
                        <div class="collapsible-body">
                            <div class="input-field">
                                <label>tPnF</label>
                                <input data-name="tPnF" class="tPnF-input morphokinetics validate" type="number" step="0.01" min="14" max="50" />
                                <span class="helper-text" data-error="Must be a number between 14-50" />
                            </div>
                            <div class="input-field">
                                <label>t2</label>
                                <input data-name="t2" class="t2-input morphokinetics validate" type="number" step="0.01" min="18" max="67" />
                                <span class="helper-text" data-error="Must be a number between 18-67" />
                            </div>
                            <div class="input-field">
                                <label>t3</label>
                                <input data-name="t3" class="t3-input morphokinetics validate" type="number" step="0.01" min="22" max="80" />
                                <span class="helper-text" data-error="Must be a number between 22-80" />
                            </div>
                            <div class="input-field">
                                <label>t4</label>
                                <input data-name="t4" class="t4-input morphokinetics validate" type="number" step="0.01" min="24" max="90" />
                                <span class="helper-text" data-error="Must be a number between 24-90" />
                            </div>
                            <div class="input-field">
                                <label>t5</label>
                                <input data-name="t5" class="t5-input morphokinetics validate" type="number" step="0.01" min="29" max="100" />
                                <span class="helper-text" data-error="Must be a number between 29-100" />
                            </div>
                            <div class="input-field">
                                <label>t6</label>
                                <input data-name="t6" class="t6-input morphokinetics validate" type="number" step="0.01" min="32" max="110" />
                                <span class="helper-text" data-error="Must be a number between 32-110" />
                            </div>
                            <div class="input-field">
                                <label>t7</label>
                                <input data-name="t7" class="t7-input morphokinetics validate" type="number" step="0.01" min="36" max="110" />
                                <span class="helper-text" data-error="Must be a number between 36-110" />
                            </div>
                            <div class="input-field">
                                <label>t8</label>
                                <input data-name="t8" class="t8-input morphokinetics validate" type="number" step="0.01" min="36" max="113" />
                                <span class="helper-text" data-error="Must be a number between 36-113" />
                            </div>
                            <div class="input-field">
                                <label>t9</label>
                                <input data-name="t9" class="t9-input morphokinetics validate" type="number" step="0.01" min="41" max="120" />
                                <span class="helper-text" data-error="Must be a number between 41-120" />
                            </div>
                            <div class="input-field">
                                <label>tM</label>
                                <input data-name="tM" class="tM-input morphokinetics validate" type="number" step="0.01" min="55" max="130" />
                                <span class="helper-text" data-error="Must be a number between 55-130" />
                            </div>
                            <div class="input-field">
                                <label>tSB</label>
                                <input data-name="tSB" class="tSB-input morphokinetics validate" type="number" step="0.01" min="74" max="140" />
                                <span class="helper-text" data-error="Must be a number between 74-140" />
                            </div>
                        </div>
                    </li>
                </ul>

                <div style="height: 36px">
                    <div id="submit-btn-${file.name}"
                        class="btn inline-block float-right"
                        onClick="submit('${file.name}')"
                    >Submit</div>
                </div>

            </div>
        </div>`;
    imagesPlaceholder.appendChild(imagePlaceholder);

    const image = imagePlaceholder.getElementsByTagName('img')[0];
    const reader = new FileReader();
    reader.onload = function (e) {
        image.setAttribute('src', e.target.result);
    };

    M.FormSelect.init(document.querySelectorAll('select'));
    M.Collapsible.init(document.querySelectorAll('.collapsible'));
    M.updateTextFields();

    reader.readAsDataURL(file);
};

function submit(imageName) {
    const card = document.getElementById(`image-card-${imageName}`);

    const eggAge = card.getElementsByClassName('egg-age-select')[0].value
      ? parseInt(card.getElementsByClassName('egg-age-select')[0].value)
      : null;
    const blastocystGradeExpansion = card.getElementsByClassName('blastocyst-grade-expansion-select')[0].value || null;
    const blastocystGradeInnerCellMass = card.getElementsByClassName('blastocyst-grade-inner-cell-mass-select')[0].value || null;
    const blastocystGradeTrophectoderm = card.getElementsByClassName('blastocyst-grade-trophectoderm-select')[0].value || null;
    const blastocystScore = card.getElementsByClassName('blastocyst-score-select')[0].value
      ? parseInt(card.getElementsByClassName('blastocyst-score-select')[0].value)
      : null;
    const morphokinetics = {};
    for (const e of card.querySelectorAll('input.morphokinetics')) {
      let value = null;
      if (e.value && e.type === 'number') {
        value = Number(e.value);
        if ((e.min && Number(e.min) > value) || (e.max && Number(e.max) < value)) {
          value = null;
        }
      }

      morphokinetics[e.dataset.name] = value;
    }

    const data = {
      eggAge,
      blastocystScore,
      blastocystGrade: blastocystGradeExpansion && blastocystGradeInnerCellMass && blastocystGradeTrophectoderm ? {
        expansion: blastocystGradeExpansion,
        innerCellMass: blastocystGradeInnerCellMass,
        trophectoderm: blastocystGradeTrophectoderm
      } : null,
      morphokinetics: Object.keys(morphokinetics).every(key => morphokinetics[key]) ? morphokinetics : null
    };

    const submitBtn = document.getElementById(`submit-btn-${imageName}`);
    submitBtn.setAttribute('disabled', 'disabled');
    [...card.getElementsByTagName('select')].forEach(x => x.setAttribute('disabled', 'disabled'));
    [...card.getElementsByTagName('input')].forEach(x => x.setAttribute('disabled', 'disabled'));
    postFormData(getFormData(currentImages[imageName], data), baseApiUrl, imageName);
};

function handleFiles(files) {
    if (files && files.length) {
        for (const file of files) {
            if (currentImages[file.name]) continue;

            if (isAnImage(file)) {
                currentImages[file.name] = file;
                clearAllButton.classList.remove('disabled');
                createImageUIFromFile(file, imagesPlaceholder);
            }
        }
    }
};

function dropHandler(event) {
    handleFiles(event.dataTransfer.files);
};

function preventDefaults (e) {
    e.preventDefault();
    e.stopPropagation();
}

const range = (start, end) => [...Array(end - start + 1)].map((_, i) => start + i);
const blastocystGradeOptions = {
  expansionOptions: ['1', '1-2', '2', '2-3', '3', '4', '5', '6'],
  innerCellMassOptions: ['A', 'A-', 'B', 'B-', 'B-/C', 'C'],
  trophectodermOptions: ['A', 'A-', 'B', 'B-', 'B-/C', 'C']
};
const blastocystScoreOptions = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17];

const form = document.getElementById('file-form');
const errorMessage = document.getElementById('error-message');
const currentImages = {};
const results = document.getElementById('results-placeholder');
const fileSelect = document.getElementById('file-select');
if (fileSelect) {
    fileSelect.value = '';
    fileSelect.addEventListener('change', function(e) {
        handleFiles(e.target.files);
    });
}

const clearAllButton = document.getElementById('clear-all-button');
if (clearAllButton) {
    clearAllButton.addEventListener('click', () => { clearAllImageCards(); } );
}

const imagesPlaceholder = document.getElementById('imageCards-placeholder');
if (imagesPlaceholder) {
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        imagesPlaceholder.addEventListener(eventName, preventDefaults, false);
    });

    ['dragenter', 'dragover'].forEach(eventName => {
        imagesPlaceholder.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        imagesPlaceholder.addEventListener(eventName, unhighlight, false);
    });

    function highlight(e) {
        imagesPlaceholder.classList.add('highlight');
    };

    function unhighlight(e) {
        imagesPlaceholder.classList.remove('highlight');
    };

}
