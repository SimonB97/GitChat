document.addEventListener('DOMContentLoaded', function() {
    var headerIcon = document.getElementById('header-icon');
    headerIcon.style.filter = 'drop-shadow(6px 5px 8px #13131393)';
});


function updateSliderValue(slider, output) {
    document.getElementById(output).innerHTML = slider.value;
}
function toggleModelNameCompressor() {
    var compressTrue = document.getElementById("compress_true");
    var modelNameCompressor = document.getElementById("model_name_compressor").parentNode.parentNode;
    if (compressTrue.checked) {
        modelNameCompressor.style.display = "inline-block";
    } else {
        modelNameCompressor.style.display = "none";
    }
}
function toggleAlphaForm() {
    var vectorStorePinecone = document.getElementById("vector_store_pinecone");
    var alphaForm = document.getElementById("alpha_form");
    if (vectorStorePinecone.checked) {
        alphaForm.style.display = "inline-block";
    } else {
        alphaForm.style.display = "none";
    }
}
$(document).ready(function() {
    toggleModelNameCompressor();
    toggleAlphaForm();

    document.getElementById("vector_store_pinecone").addEventListener("change", toggleAlphaForm);
    document.getElementById("vector_store_chromadb").addEventListener("change", toggleAlphaForm);
    document.getElementById("vector_store_deeplake").addEventListener("change", toggleAlphaForm);

    $('.info-icon').qtip({
        content: {
            attr: 'data-tooltip'
        },
        style: {
            classes: 'qtip-bootstrap'
        },
        position: {
            my: 'top left',
            at: 'bottom right',
            adjust: {
                method: 'flip (invert adjust.x/y)'
            },
        },
        show: {
            // on click
            event: 'click',
            effect: function() { $(this).fadeIn(250); },
            solo: true
        },
        hide: {
            event: 'unfocus click',
            effect: function() { $(this).fadeOut(250); }
        },
    });
});