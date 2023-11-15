// script.js
document.addEventListener('DOMContentLoaded', function() {
    // Listen for form submission
    const recommendationsForm = document.querySelector('form[action="/search"]');
    console.log(recommendationsForm)
    recommendationsForm.addEventListener('submit', function(event) {
        event.preventDefault();
        
        // Get user input
        const songNameInput = document.getElementById('song_name');
        const yearInput = document.getElementById('year');
        const songNameHidden = document.getElementById('song_name_hidden');
        const yearHidden = document.getElementById('year_hidden');
        
        // Set the values of the hidden fields
        songNameHidden.value = songNameInput.value;
        yearHidden.value = yearInput.value;

        // Submit the form
        recommendationsForm.submit();
    });
});

function updateSliderValue() {
    var slider = document.getElementById("valueSlider");
    var displayValue = document.getElementById("selectedValue");
    displayValue.innerText = slider.value;
}