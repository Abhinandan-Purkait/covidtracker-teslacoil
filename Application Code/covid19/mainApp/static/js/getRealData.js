const selectElement = (name) => {
    return document.querySelector('#' + name);
};

async function getRealData() {
    const url = 'https://covid19.mathdro.id/api';

    let data = await (await fetch(url)).json();

    selectElement('confirm-info').innerHTML = data.confirmed.value;
    selectElement('recover-info').innerHTML = data.recovered.value;
    selectElement('death-info').innerHTML = data.deaths.value;
    selectElement('last-update').innerHTML = new Date(data.lastUpdate).toUTCString();
}

async function getAllCountries() {
    const url = 'https://covid19.mathdro.id/api/countries';

    let data = await (await fetch(url)).json();

    data.countries.forEach((country) => {
        var option = document.createElement("OPTION");
        option.innerHTML = country.name;
        option.value = country.name;
        selectElement('country-picker').appendChild(option);
    });
}

async function countryChoose() {
    let country = selectElement('country-picker').value;

    if (country === 'global') {
        getRealData();
    } else {

        const url = `https://covid19.mathdro.id/api/countries/${country}`;

        let data = await (await fetch(url)).json();

        selectElement('confirm-info').innerHTML = data.confirmed.value;
        selectElement('recover-info').innerHTML = data.recovered.value;
        selectElement('death-info').innerHTML = data.deaths.value;
        selectElement('last-update').innerHTML = new Date(data.lastUpdate).toUTCString();
    }
}

getRealData();
getAllCountries();