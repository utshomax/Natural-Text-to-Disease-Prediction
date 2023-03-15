
def getWords(paragraph):
    sentence = paragraph.split(".")
    words = [sentence[i].split(" ") for i in range(len(sentence))]
    #flatten the words list
    word = [item for sublist in words for item in sublist]
    print("words", word)
    return word


def getSimilarity(word1, word2):
    # Convert words to lowercase and remove non-alphabetic characters
    word1 = ''.join(filter(str.isalpha, word1.lower()))
    word2 = ''.join(filter(str.isalpha, word2.lower()))
    
    # Convert words to sets of characters
    set1 = set(word1)
    set2 = set(word2)
    
    # Calculate intersection and union of sets
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    
    # Calculate Jaccard similarity coefficient
    similarity_score = len(intersection) / len(union)
    
    # Return similarity score as percentage
    return similarity_score * 100

def similarity(word1, word2):
    """
    This function takes in two words as input and calculates their similarity score,
    taking into account the displacement of characters in a word and also subsequent
    same index as a better option.
    """
    # Convert the words to lowercase
    word1 = word1.lower()
    word2 = word2.lower()

    # Get the lengths of the words
    len1 = len(word1)
    len2 = len(word2)

    # Calculate the maximum possible similarity score
    max_score = max(len1, len2)

    # Initialize the similarity score to 0
    score = 0

    # Keep track of the previous match index for each word
    prev_match1 = -1
    prev_match2 = -1

    # Iterate through the characters of the first word
    for i, c1 in enumerate(word1):
        # Check if the character is also in the second word
        if c1 in word2:
            # Get the index of the character in the second word
            j = word2.index(c1)

            # Calculate the displacement score
            displacement = abs(i - j)

            # Check if the character is at a subsequent index in both words
            if i > prev_match1 and j > prev_match2:
                # Add a bonus score for a subsequent match
                score += max_score - displacement + 1
            else:
                # Add the displacement score to the similarity score
                score += max_score - displacement

            # Update the previous match indices
            prev_match1 = i
            prev_match2 = j

            # Remove the character from the second word to avoid counting it again
            word2 = word2[:j] + word2[j+1:]

    # Return the similarity score
    return score


def classifySymptoms(words):
    # Read symptoms from file
    with open("symptoms.txt", "r") as file:
        symptoms_list = file.read().split(',')
    
    symptomsp = []

    for i in range(len(words)):
        # Convert symptoms to set of characters
        symptoms_set = set(words[i])
        
        # Calculate intersection of symptoms with each disease
        intersection = [symptoms_set.intersection(set(symptoms_list[i])) for i in range(len(symptoms_list))]
        
        # Calculate union of symptoms with each disease
        union = [symptoms_set.union(set(symptoms_list[i])) for i in range(len(symptoms_list))]
        
        # Calculate Jaccard similarity coefficient for each disease
        similarity_score = [len(intersection[i]) / len(union[i]) for i in range(len(symptoms_list))]
        #n_similarity_score = [similarity(words[i], symptoms_list[i]) for i in range(len(symptoms_list))]
        # Return disease with highest similarity score
        print("Similarity score: ", max(similarity_score))
        if(max(similarity_score) < 0.5):
            continue
        jsim = symptoms_list[similarity_score.index(max(similarity_score))]
        #nsim = symptoms_list[n_similarity_score.index(max(n_similarity_score))]

        symptomsp.append(jsim)
    return symptomsp

def main():
    paragraph = input("Enter a paragraph: ")
    words = getWords(paragraph)
    # print("Number of words: ", len(word))
    # print("Number of sentences: ", len(sentence))
    #print("Jaccard similarity: ", getSimilarity(word[0], word[1]))
    #symptoms = input("Enter symptoms: ")
    res = classifySymptoms(words)
    print("Symptoms: ", res)
    # print("Jaccard similarity: ", cs)
    # print("New similarity: ", ns)
main()