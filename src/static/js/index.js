$(function () {
    // Create a Showdown converter
    var converter = new showdown.Converter();

    $('#chatbot-form-btn').click(function (e) {
        e.preventDefault();
        $('#chatbot-form').submit();
    });
    $('#chatbot-form-btn-clear').click(function (e) {
        e.preventDefault();
        $('#chatPanel').find('.media-list').html('');
        
        // Send an AJAX request to the /clear_memory route
        $.ajax({
            type: "POST",
            url: "/clear_memory",
            success: function (response) {
                console.log("Memory cleared successfully");
            },
            error: function (error) {
                console.log("Error clearing memory:", error);
            }
        });
    });

    // Add event listener for keypress on the textarea
    $('#text_input').on('keypress', function (e) {
        if (e.which === 13 && !e.shiftKey) { // Check if Enter key is pressed and Shift key is not pressed
            e.preventDefault();
            $('#chatbot-form').submit();
        }
    });

    $('#chatbot-form').submit(function (e) {
        e.preventDefault();
        var message = $('#text_input').val();

        // Check if the input is empty to avoid sending an empty request
        if (message.trim() === '') {
            return;
        }

        $(".media-list").append(
            '<li class="media"><div class="media-body"><div class="media"><div style = "text-align:right; color : #f0f0f0; background-color: #444654" class="media-body">' +
            message + '</div></div></div></li>');

        // Clear the text field immediately after submission
        $('#text_input').val('');

        $('#circleG').show();

        $.ajax({
            type: "POST",
            url: "/",
            data: {text_input: message}, // Pass the saved message as data
            success: function (response) {
                var answer = response; // Change this line

                // Convert the response to HTML using Showdown
                var answerHtml = converter.makeHtml(answer);

                // Create a temporary element to hold the HTML content
                const tempEl = document.createElement('div');
                tempEl.innerHTML = answerHtml;

                // Apply syntax highlighting to code blocks
                tempEl.querySelectorAll('pre>code').forEach(codeElement => {
                    hljs.highlightElement(codeElement);
                });

                // Retrieve the updated HTML content
                answerHtml = tempEl.innerHTML;

                $(".media-list").append(
                    '<li class="media"><div class="media-body"><div class="media border"><div style = "color : #f0f0f0; background-color: #343541" class="media-body">' +
                    answerHtml + '</div></div></div></li>');
                setTimeout(function() {
                    $(".fixed-panel").scrollTop($(".fixed-panel")[0].scrollHeight);
                }, 100);

                $('#circleG').hide();
            },

            error: function (error) {
                console.log(error);
                $('#circleG').hide();
            }
        });
    });
});